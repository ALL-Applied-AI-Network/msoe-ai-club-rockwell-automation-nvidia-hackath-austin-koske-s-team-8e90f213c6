# Operator-in-the-loop scope clarification

Recorded October 21, 2026 from a reply Austin supplied in chat. The sender's name
and the original message timestamp were not supplied. This is project-specific
guidance and supersedes the earlier public-document audit's open question about
direct station integration during the hackathon.

## Confirmed scope from the reply

- Our system cannot directly communicate with existing line sensors or actuators.
- Build an interface that an operator can respond to.
- Teledyne uses a static color database that cannot be updated.
- The existing sorting decision is not directly linked to the intended recipe.
- PLC integration can be pursued after the hackathon, once the operator-mediated
  approach has been proven. No hackathon OPC UA/PLC adapter is part of this scope.

The reply describes almost all colors going to the location labeled good nearest
the line, and matches to the database's color/weight parameters going to the
location labeled scrap. This differs from the earlier public guide. Preserve that
description and confirm the exact destination semantics on site. Do not infer
recipe correctness from an existing route or silently reverse the labels.

## Consequences for the architecture

The physical MVP uses CSI-approved video, operator-supplied intended recipe/batch
context, a confirmed physical vial association and an operator confirmation of
dispensing completion. It can compare a completed fill with a separately verified
acceptable endpoint reference under a compatible camera/measurement profile.
Operator entry must retain reviewer, time and evidence; it is not a sensor reading.

Do not assume live flow-meter/phase/completion tags. During-fill metered-reference
comparison needs separately supplied, qualified progress evidence and remains
unavailable when that evidence is missing. Image detection/masking can continue;
planned recipe quantities and elapsed time do not become measured delivery.
CSI-provided recording/log exports could support later offline analysis, if
available. Their availability is not confirmed by the reply.

The UI should present intended recipe, observed evidence, assessment/reason,
uncertainty and a review request. Log the operator's acknowledgement, actual action
and outcome separately. Use CSI's approved handling workflow; the UI itself does
not hold equipment, reroute a vial, change inspection thresholds or execute a flush.

Show recipe assessment, reported station route, physical fixture label and actual
operator disposition as distinct fields. Keep route mapping unconfirmed until
CSI verifies it. Model agreement and the good/scrap destination are not independent
product-quality truth.

The full phase/metered trajectory and actuation loop remains useful in simulation.
Demonstrate simulated actions as simulated; preserve private truth for evaluation.
The physical demonstration is operator-mediated. Measure warning availability,
operator response time, recorded action and independent outcome, plus unavailable
cases. Do not present a UI acknowledgement as a machine acknowledgement.
