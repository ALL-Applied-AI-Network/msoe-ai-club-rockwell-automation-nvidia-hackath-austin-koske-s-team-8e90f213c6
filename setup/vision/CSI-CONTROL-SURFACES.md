# What Part 2 can act on at CSI

## Confirmed hackathon boundary — supersedes the open interface questions below

Austin relayed a project-specific reply: **no direct communication with existing
line sensors or actuators; use an operator interface. Teledyne's static database
cannot be updated. PLC integration is a post-hackathon possibility.** The HMI/OPC
UA audit below identifies equipment capabilities and possible future integration,
not accessible hackathon controls. Do not build a station adapter or database
update workflow for this visit. See the
[recorded clarification and input/route implications](../../team_handoff/part2/OPERATOR-CLARIFICATION.md).

The reply describes most colors going to the location labeled good and database
color/weight matches going to the location labeled scrap. That differs from the
earlier guide. Keep route, physical fixture label, independent recipe assessment
and operator disposition separate; confirm the route semantics on site.

Reviewed October 21, 2026 against the public competition repository at commit
`fc51de15e4bd8dae42f362c6045cffd106b8727a`. This is a documentation and screenshot
audit, not a connection test. No station access, writable tag or command endpoint
has been verified. The existing Part 2 brief remains a valid advisory architecture;
this audit identifies concrete places where CSI could connect its outputs.

## Findings that change the implementation plan

**The most concrete first action is a recipe-transition review before batch
release by an operator.** The published Batch Scheduler screenshot shows recipe selectors,
`Release to Production`, `Clear Queue`, `Recipe Setup` and batch history.
Our application can present a transition-risk assessment before an operator
uses the existing release workflow. The subsequent reply rules out an external
line-control connection during the hackathon. A screenshot is evidence of
an operator control, not evidence of a callable API or team permission.
[Published scheduler image](https://msoe-ai-club.github.io/maic-content/images/article_content/rockwell-nvidia-fall-2026/fluid-batch-scheduler.webp).

**Recipe setup exposes inspection settings as well as liquid quantities.** The
image includes per-tank mL fields, upper/lower inspection weight limits and
per-channel RGB upper/lower bounds under `Vial Inspect Color`. This makes
recipe-specific inspection configuration a concrete subject for CSI to explain.
The later reply confirms the Teledyne database is immutable for this project.
The image does not establish that these fields drive the current Teledyne decision,
are loaded automatically for every batch, or can accept our model output.
[Published recipe-setup image](https://msoe-ai-club.github.io/maic-content/images/article_content/rockwell-nvidia-fall-2026/fluid-recipe-screen.webp).

## Documented surfaces and future possibilities

The physical hackathon output is the operator interface/log. All external control
interfaces in the final column are future possibilities, outside confirmed scope.

| Surface | What is documented | What we can build now | What physical integration still needs |
| --- | --- | --- | --- |
| Our operator display and evidence log | Owned by our application; no plant configuration required for local file inputs | Recipe/phase/progress view, explicit unavailable state, transition warning, per-vial review request and operator receipt history | Approved live inputs if testing from the running station; this does not itself stop or reroute anything |
| Batch scheduling and release | Recipe selectors, queue, release, clear and history are visible in the HMI | Rank approved A→B transitions by predicted risk; recommend review before release; record the operator's decision | CSI confirmation of release/queue semantics, accepted production-order constraints and any approved gate/request interface |
| Recipe quantities | Per-tank volume fields and recipe save/load controls are visible | Simulate approved recipe/conditioning alternatives and produce an operator proposal | Plant-manager guidance; no team changes to recipes or online dose correction are authorized by this audit |
| Existing inspection configuration | HMI shows RGB bounds, weight limits and station/inspection requirement flags | Compare our intended-recipe assessment against the recorded existing verdict; investigate compatible reference/configuration proposals | CSI explanation of the RGB/database relationship, units/measurement location, write permissions and validation procedure; do not bypass inspection |
| Good/scrap routing | Dataset 1 documents the existing inspection-to-good/scrap fixture path | A physical-vial-specific review/diversion request, plus a simulated routing result | Confirmed vial association, decision deadline, approved override/request point and execution receipt; no external routing override is published |
| Stop/hold/continue | No specific team-accessible hold or resume interface is published | Local latched review state and a proposed operator workflow; simulated pause/resume | CSI-defined safe operational boundary and an approved method; clearing the queue is not evidence of a safe pause or emergency-stop command |
| Flush, drain or prime | Organizer feedback supplied by Austin confirms a flush drain; the public pages do not specify its control interface or an approved cleaning sequence | Test tagged simulated cleaning policies and log a proposed CSI procedure/receipt | Exact procedure, destination, settings, operator authority and completion evidence; an acknowledgement alone is not physical recovery |
| Digital twin | Supplied station USD/rendering assets and CSI's Emulate3D/Omniverse workstation are documented | Script scenarios, change simulated process conditions and test advisory-to-simulated-action behavior | A complete simulated carrier/response loop remains implementation work; supplied USD assets do not establish a PLC-connected operational twin |

The [fluid-station guide](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing.md)
documents recipe semantics; the
[final-inspection guide](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Vial-Inspection/Rockwell-NVIDIA-Fall-2026-Vial-Inspection.md)
documents the good/scrap path. The
[collection/testing guide](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Data-Collection/Rockwell-NVIDIA-Fall-2026-Data-Collection.md)
requires scheduled supervision and explicit approval before touching equipment;
it prohibits independent PLC/controller/FactoryTalk/recipe changes and unapproved
industrial-network connections. Physical permissions are not implied by a
technical feature being present.

## Operating details the twin and proposed actions must respect

The fluid-station page specifies a maximum total vial volume of 80 mL, practical
minimum 10 mL for a selected component, at most 50 mL from T-200, and no T-300
use because of its configuration fault. Dispense order is T-200 → T-400 → T-100,
skipping unused tanks. The stored tank identity is not a stable color identity:
colors can change. Recipe changes require the plant manager's guidance.
[Equipment and recipes](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing/Rockwell-NVIDIA-Fall-2026-Fluid-Mixing.md).

Consequences for our design:

- Optimize a proposed **sequence of approved recipes** only within production-order
  constraints CSI confirms. Do not treat ingredient order as a free control variable.
- An approved clear-only conditioning step is a question for CSI, not a documented
  flush command. A vial fill and a drain flush are different operations.
- The earlier simulated 8 mL prime-to-drain fixture is a model experiment, not an
  executable station recipe. It cannot be translated directly into a selected
  recipe component under the documented practical 10 mL minimum.
- The page describes three stored/selectable recipes; its scheduler image shows
  ten queue rows. Confirm the distinction between recipe slots and queued batches
  rather than hard-coding a three-entry queue limit.
- `Release to Production` is a batch-scheduler control. Do not interpret that label
  as the final acceptance/release of an inspected vial.

## What the linked integration tools actually provide

The [resources page](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Resources/Rockwell-NVIDIA-Fall-2026-Resources.md)
points to FactoryTalk Optix/OPC UA and NVIDIA's Factory Operations FOX architecture.
It explicitly says FOX is architecture context rather than a complete testbed
implementation, and FactoryTalk/OPC UA access needs approval.

[Rockwell's OPC UA documentation](https://www.rockwellautomation.com/en-us/docs/factorytalk-optix/1-00/contents-ditamap/using-the-software/opc-ua.html)
describes publishing information-model nodes and client/server reads, writes and
method invocation. Its [client reference](https://help.optix.cloud.rockwellautomation.com/1.4.1.7/en/Reference/FTOptix-OPCUAClient/ObjectTypes/OPCUAClient.html)
lists subscriptions, value reads/writes and method calls. These are generic
capabilities, not evidence that CSI runs an accessible server, exposes a reject
tag, permits a write, or uses this exact Optix configuration. Endpoint, node/tag
map, identity permissions, timestamp semantics and command acknowledgement must
come from CSI. No default/sample vendor endpoint is a CSI endpoint.

NemoClaw/NeMo Agent Toolkit can organize software tools and workflows. They do not
supply the missing station adapter or authorize plant commands. The
[FOX reference](https://blogs.nvidia.com/blog/factory-operations-fox-blueprint-ai-brain/)
can inform a later orchestration layer; the current baseline can use deterministic
evidence and state logic without adding a language model to command selection.

Dataset 3's FANUC ER-4iA/gripper and its linked
[ROS 2 driver](https://fanuc-corporation.github.io/fanuc_driver_doc/main/docs/quick_start/quick_start.html)
are a **separate cell**. They are not an established actuation interface for our
vial line. Its [dataset guide](https://github.com/MSOE-AI-Club/maic-content/blob/main/unlisted/Rockwell-NVIDIA-Fall-2026-Grasping-Sorting/Rockwell-NVIDIA-Fall-2026-Grasping-Sorting.md)
also requires plant-manager/mentor involvement before physical commands.

## Concrete Part 2 output contract to discuss with CSI

The following are proposed logical messages, **not existing PLC tag names**:

| Proposed output | Evidence and intended effect |
| --- | --- |
| Transition review request | Previous/next recipe, source/context/reference IDs, estimated risk or unavailable reason. Operator reviews before batch release. |
| Vial review request | Physical vial ID, intended recipe, source/decision time, persistent mismatch evidence and current deadline. Operator or an approved adapter handles review/disposition. |
| Approved-recovery proposal | CSI procedure ID and evidence that motivated it. CSI accepts or rejects the proposal, then records actual execution separately. |
| Recovery verification | Assessment of distinct subsequent completed vials and independent outcomes; proposes resumption only through the agreed workflow. |

For every action, separate request, acceptance, execution and verification. Record
the request ID, associated batch/vial, operator/adapter, native clock mapping,
expiry/deadline, acknowledgement or rejection, actual completion and resulting
disposition. Reject stale/wrong-vial requests. The local unavailable/review latch
does not establish an equipment hold.

Our next real demonstration can be **human-mediated closed-loop operation**:
observe a supervised physical fill, show a qualified transition/vial review
request, have CSI use its existing approved workflow, then verify/log what
happened. Record the intervention and measure time to action. If no qualified
recipe comparison or unacceptable vial is available, report that limit. An
external automated closed loop is a further integration task after CSI supplies
and approves its actual interface.

## Remaining questions for the operator-mediated visit, prepared but not sent

1. Where and when can an operator respond to our alert using an existing approved
   workflow? How should they identify the exact physical vial and record the action?
2. Please confirm which physical destination receives database color/weight
   matches, and which receives other colors; how do the good/scrap labels relate?
3. What independent intended-recipe reference and outcome assessment can CSI
   provide, given that the existing static inspection is not recipe-aware?
4. What CSI-performed recovery procedure may be observed and logged, if warranted?
5. What video files or separately approved video setup can we use without direct
   existing-sensor access? Can CSI supply log exports for offline analysis? Such
   exports are optional; a live meter feed is not assumed.

## Audit scope and retained evidence

Read the main overview and all seven event subpages: Fluid Mixing, Vial Inspection,
Grasping/Sorting, Data Collection, Resources, Compute and Coding Agents. Followed
control-related vendor references (Optix OPC UA, FOX, NeMo/NemoClaw, SOP monitoring
and FANUC), and visually inspected both published fluid-station HMI images.
General training/model/tutorial links are learning references, not a published
station I/O contract; this audit does not claim an exhaustive recursive read of
every vendor site's documentation.

The `msoe-maic.com/library` article links returned 404, so their source Markdown
was read directly from the linked public GitHub repository. The Drive twin link
was not accessible through the browser tool; it was not downloaded again. Some
Optix tutorial content was only partly exposed; the accessible official node/
client references above were used instead. No absent material was assumed to
contain writable tags.

Local evidence: `C:/Users/Austin/ai-club-tools/results/csi-control-audit-20261021/source-v1/`.
It contains eight immutable Markdown snapshots, source URLs/hashes and two HMI
images with hashes. Existing datasets, source labels, model settings, Part 2
sharing packages and plant controls remain unchanged. No message was sent and
no main-branch publication was performed.
