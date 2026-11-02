# The industry problem: wrong product after a changeover

Research for the talk and slides (2026-10-30). Each figure has its source and how much to trust
it. **Independent** means peer-reviewed, government, or a named study by an industry body or
insurer. **Vendor** means marketing material from someone selling a fix; use it only as "can
cost".

## The one-sentence problem

Lines change products more and more often. The first units after a change can still hold the
previous product. Today's checks either don't know which product was ordered, or catch the problem
after the product has moved on.

## Headline numbers

| Claim | Figure | Source | Trust |
| --- | --- | --- | --- |
| Cost of poor quality | about **15–20% of sales**, up to 40% at the worst performers | ASQ, as quoted by [ReliaMag](https://reliamag.com/articles/cost-of-poor-quality-scrap-rework-returns/); [IISE](https://iise.org/details.aspx?id=22118): about 15% average, 5–35% range | Independent (IISE), ASQ-attributed |
| Food and beverage recall | about **$10M direct cost** on average | FMI/GMA study, via [Food Safety](https://www.food-safety.com/signature-series/recall-the-food-industrys-biggest-threat-to-profitability/) | Industry study (2011, dated) |
| Significant recall, all sectors | about **$12M** on average (food and beverage about $9.5M), 2012–2017 | [Allianz via Business Insurance](https://www.businessinsurance.com/Product-recalls-cost-companies-$12-million-on-average-Allianz-/), [FoodNavigator](https://www.foodnavigator.com/Article/2017/12/13/Product-recall-risks-growing-in-size-and-number-Allianz) | Independent (insurer claims data) |
| Top cause of US food recalls | **undeclared allergens, 38.8%** of FDA and USDA recalls in 2025; 105 of 266 FDA recalls in the first half of 2026 | [Statista/PIRG](https://www.statista.com/chart/36281/fda-and-usda-food-recalls-in-the-united-states/), [Food Processing](https://www.foodprocessing.com/food-safety/regulatory-compliance/news/55399686/food-recalls-fell-in-the-first-half-of-this-year) | Independent |
| Unplanned downtime | Fortune Global 500 lose **$1.4T a year, 11% of revenue**; automotive **$2.3M per hour** | [Siemens True Cost of Downtime 2024](https://assets.new.siemens.com/siemens/assets/api/uuid:1b43afb5-2d07-47f7-9eb7-893fe7d0bc59/TCOD-2024_original.pdf) | Named industry study |
| Human visual inspection | **80% accuracy** is the cited industry average; inspectors miss **20–30%** of defects | [Sandia study, *Human Factors* 2015](https://api.crossref.org/works/10.1177%2F0018720815602389); [Landing AI](https://landing.ai/news/giving-weary-human-eyes-a-rest-improving-accuracy) citing Sandia | Peer-reviewed (benchmark cited in the paper) |
| Inspector pay | **$48,570** median a year for quality control inspectors (SOC 51-9061), May 2025 | [BLS Occupational Outlook Handbook](https://www.bls.gov/ooh/production/quality-control-inspectors.htm) | Government |
| Edge computer | NVIDIA DGX Spark (GB10): 64 GB from **$4,999**; 128 GB about **$6,950** list (launched at $3,999, raised with memory prices) | [HotHardware](https://hothardware.com/news/nvidia-dgx-spark-64gb-release), [ServeTheHome](https://servethehome.com/nvidia-dgx-spark-64gb-launched-and-big-128gb-gb10-price-increases) | Trade press; check a reseller before quoting |
| Labour | **3.8M** US manufacturing jobs needed by 2033; **1.9M could go unfilled** | [Deloitte and Manufacturing Institute, 2024](https://www2.deloitte.com/us/en/pages/about-deloitte/articles/press-releases/us-manufacturing-could-need-new-employees-by-2033.html) | Independent |
| AI for quality | **Quality control is the #1 AI use case** (50% plan to use it), two years running; 95% investing in AI | [Rockwell Automation, State of Smart Manufacturing 2025](https://plex.rockwellautomation.com/content/dam/plex/legacy/2025-06/state-of-smart-manufacturing-2025.pdf) | Sponsor's own survey (1,500+ manufacturers) |
| Machine vision market | **$7–18B in 2026**, depending on scope; quality inspection is the main driver | [GMI](https://www.gminsights.com/industry-analysis/industrial-machine-vision-market), [Fortune Business Insights via GII](https://www.giiresearch.com/report/fbs2020111-machine-vision-market-size-share-growth-global.html) | Market research |

## Why changeovers are where it goes wrong

- **More changeovers:** SKU proliferation means shorter runs and more changeovers
  ([MRPeasy](https://mrpeasy.com/blog/sku-rationalization)). Rockwell notes that most packaging lines
  weren't built for rapidly changing demand
  ([Rockwell Automation](https://www.rockwellautomation.com/en-in/company/news/magazines/how-ecommerce-is-impacting-packaging-equipment-design-.html)).
- **Flushes are expensive and blunt:** a hot-water flush and recharge can stop a line for
  **20–30 minutes** per changeover, using a lot of water and product
  ([US patent 8430111](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8430111)). One pigging
  vendor reports **500–600 lb of product wasted per changeover** before its fix
  ([HPS via Fluid Handling Pro](https://fluidhandlingpro.com/news/flushing-your-pipelines-heres-why-pigging-technology-is-the-better-choice-for-food-and-beverage-manufacturers/); vendor).
  Tetra Pak names mix-phase changeovers as a product-loss point
  ([Tetra Pak](https://www.tetrapak.com/insights/cases-articles/where-are-energy-water-and-product-being-lost-in-your-operation)).
- **Carryover is the mechanism behind the top food recall cause:** an undeclared allergen is often
  the previous product's residue in the next one (cross-contact). Linewatch measures colour, so it
  can't detect allergens. Its point is that the same changeover failure is visible and
  catchable when the products differ in colour.
- **In pharma, carryover can be total:** in finished-dose manufacturing, residue on equipment may
  be *100% carried over* into the next product (APIC cleaning-validation guidance, via
  [ComplianceOnline](https://www.complianceonline.com/resources/downloads/Cleaning-Validation-API-Plants.pdf)).
  Product mix-ups are treated as the most acutely dangerous recall type
  ([RecallCheck](https://recallcheck.net/drug-recalls/reasons)), e.g. a 2026 Class II recall of
  6,143 bottles for a tablet mix-up. Batch values of **$0.5M to several million** are vendor
  estimates ([iFactory](https://ifactoryapp.com/industries/manufacturing-plant/ai-pharma-batch-consistency-manufacturing)).
- **Measurement often comes too late:** many plants still pull manual samples, and the lab
  "often can't keep up or ends up correcting product that has already moved past the measurement
  point" ([Endress+Hauser application note](https://bdih-download.endress.com/file/005056A5E3831EDBB8DDE5A4AA14513E/CS0362%20AF22%20inline%20color%20monitoring.pdf)).
  Inline colour sensors exist, but they measure colour against a fixed spec; they don't check
  which product was ordered for that unit.
- **Paints and coatings:** colour shifts come from raw materials, equipment and operators, and
  visual matching varies between evaluators. Off-colour batches waste raw material and cost money
  to dispose of ([Datacolor](https://datacolor.com/?p=178186), [PCI](https://www.pcimag.com/articles/82864-color-technology-and-today-8217-s-high-performance-cost-containment-conundrum); vendor and trade press).

## Where Linewatch's idea applies

| Industry | Changeover risk | What a recipe-aware camera adds |
| --- | --- | --- |
| Beverages and syrups | Flavour and colour changes on shared fillers; flush water and product loss | Flags the off-colour first units; flushes only when needed |
| Pharma fill-finish (liquids) | Mix-ups and cross-contamination; very costly batches | Per-unit check against the ordered product, with an audit trail |
| Paints and coatings | Tint changes on shared lines; off-colour batches | Catches the off-colour unit at fill, not at the customer |
| Cosmetics and personal care | Many shades on one line | Shade checked against the order per unit |
| Chemicals | Dosing errors and changeover residue | STOP & CHECK DOSING after persistent deviation |

## What we measured on CSI's line

- In Run 5, **1 in 3 vials** was the first of a new recipe. **All 5** of those that could be
  measured were off recipe, and **all 26** vials that reached the trays went to the Good tray
  ([evaluation](evaluation.md#line-runner-on-run-5-v1)).
- CSI's own flush clips show a water flush holding *more* dye than the vial before it: the line
  lags about one vial, so **one flush doesn't clear a change**
  ([evaluation](evaluation.md#csi-capture-water-flushes-after-red-and-yellow-2026-10-28-clips-measured-2026-10-30)).

## Talk lines (each one is sourced above)

- "Poor quality costs manufacturers 15 to 20 percent of sales."
- "A food recall costs about ten million dollars, and the top cause is the previous product
  ending up in the next one."
- "Human inspectors miss 20 to 30 percent of defects, and there are 1.9 million manufacturing jobs
  we may not be able to fill."
- "Rockwell's own survey: quality control is manufacturers' number-one use for AI."
- "On CSI's line, one in three vials was the first after a recipe change, and every one we could
  measure was the wrong colour, and went to the Good tray."

Don't put a CSI dollar-savings figure on a slide: we have no CSI volumes or unit costs.
