#!/usr/bin/env bash
# Back up a CSI session folder to a DRAFT GitHub release (repo writers only; never published).
# Only upload data CSI has approved for off-site storage. Keep the USB copy regardless.
#
#   bash setup/gb10/upload-session.sh /path/to/csi-20261027/session-01 [--tag NAME] [--dry-run]
#
# Writes SHA256SUMS.txt inside the folder (if missing), packs it as an uncompressed tar split
# into <1.9 GB parts, and uploads parts + checksums. Restore: cat NAME.tar.part-* > NAME.tar
# && sha256sum -c NAME.parts.sha256 && tar -xf NAME.tar
set -euo pipefail

DIR="${1:-}"; shift || true
TAG=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[ -d "$DIR" ] || { echo "Usage: upload-session.sh SESSION_DIR [--tag NAME] [--dry-run]" >&2; exit 2; }
DIR="$(cd "$DIR" && pwd)"
NAME="$(basename "$DIR")"
[ -n "$TAG" ] || TAG="csi-data-$(date +%Y%m%d)-$NAME"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" remote get-url origin)"
command -v gh >/dev/null && gh auth status >/dev/null 2>&1 || { echo "gh is not installed/authenticated; keep the USB copy and run this from a logged-in laptop" >&2; exit 1; }

if [ ! -f "$DIR/SHA256SUMS.txt" ]; then
  (cd "$DIR" && find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt)
fi
echo "$(wc -l < "$DIR/SHA256SUMS.txt") files, $(du -sh "$DIR" | cut -f1) in $DIR"

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
tar -cf - -C "$(dirname "$DIR")" "$NAME" | split -b 1900m -d -a 3 - "$WORK/$NAME.tar.part-"
(cd "$WORK" && sha256sum "$NAME".tar.part-* > "$NAME.parts.sha256")
cp "$DIR/SHA256SUMS.txt" "$WORK/$NAME.files.sha256"
ls -la "$WORK"

NOTES="CSI session data backup ($NAME). DRAFT: do not publish. Upload approved by CSI for off-site storage.
Evaluation-only real data: no training or tuning. Restore:
cat $NAME.tar.part-* > $NAME.tar && sha256sum -c $NAME.parts.sha256 && tar -xf $NAME.tar && (cd $NAME && sha256sum -c SHA256SUMS.txt)"
if [ "$DRY" = 1 ]; then echo "[dry run] would create draft release $TAG on $REPO with the files above"; exit 0; fi

if gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
  echo "Release $TAG exists; adding/replacing assets"
else
  gh release create "$TAG" -R "$REPO" --draft --title "CSI data backup: $NAME" --notes "$NOTES"
fi
gh release upload "$TAG" -R "$REPO" --clobber "$WORK"/*
echo "Uploaded to draft release $TAG. Verify: gh release download $TAG -R $REPO -D <dir>"
