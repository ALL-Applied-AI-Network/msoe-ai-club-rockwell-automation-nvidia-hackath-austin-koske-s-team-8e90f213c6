# Original briefs

These documents shipped with the October 20 data release:

- [Architecture implementation brief](Team-Architecture-Implementation-Brief.pdf)
- [Data guide](Team-Data-Guide.pdf)
- [Original architecture diagram](original-architecture.png)

They describe the original YOLO → R-CNN → SAM workstreams. The maintained
[architecture](../architecture.md) uses direct temporal U-Net, and the current
[data policy](../data.md#training-and-evaluation-policy) requires synthetic
CSI-domain fitting. The older training instructions and assignments are history.

The [October16 tooling catalog](resources-20261016.md) records earlier optional-tool research; it is not the runtime architecture.

The original release attachments remain unchanged. Dated development notes and
branch reviews are available in Git history; verification JSON remains under
`setup/verification/`.
