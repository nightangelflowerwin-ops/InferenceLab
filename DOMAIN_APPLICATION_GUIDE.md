# Where Relay fits: domain applications and GTM engineering

Relay demonstrates a useful building block: several worker processes share durable jobs, produce reports and recover unfinished work after a worker crashes. The same pattern can support many business workflows.

Think of it as a reliable workroom. The queue is the tray of incoming work. Workers are the people carrying out tasks. The saved job record tells us what is waiting, what finished and what needs another attempt.

**Shipped today:** text-file uploads, word-count reports, separate worker processes, durable local jobs, result downloads and demonstrated crash recovery. **The applications below are proposed extensions.** Relay does not currently connect to a CRM, clinical system, payment service or email provider. It runs on one laptop; its database is not replicated across machines.

## What a GTM engineer does

Go-to-market means helping a product reach customers, turn interest into revenue and keep customers receiving value. A GTM engineer builds the connections and automation behind that journey.

For example: a prospect submits a form, their company record is checked, the inquiry reaches the appropriate sales team, follow-up is prepared, and the result appears in a report. The engineer makes that chain work reliably and makes its failures visible.

Queues help when work takes time, arrives in bursts or depends on another service. They do not need to sit between every tiny operation. A simple immediate response can stay simple.

## Across domains

Every example here describes an administrative or commercial workflow, rather than a claim that the domain application is already built.

| Domain | Practical application | Work the queue would carry | Result a person can review | GTM connection |
| --- | --- | --- | --- | --- |
| Clinical and healthcare operations | Appointment and intake worklist | Validate appointment records, flag missing administrative fields, prepare permitted reminder drafts | Missing-information list, scheduling conflicts and delivery-status report | Sell to clinic operators; measure staff time, completion and appointment workflow reliability |
| Healthcare software and medical-device businesses | Account onboarding and sales handoff | Check clinic-account records, route implementation requests, track document completion | Account readiness and handoff report | Help hospital or clinic customers evaluate, buy and adopt the product |
| B2B SaaS | Inbound lead-to-sales pipeline | Clean form submissions, group accounts, apply declared qualification rules, suggest owners | Reviewed lead queue with reasons and source links | Demand generation, sales operations and activation |
| Ecommerce and retail | Order exception and retention worklists | Check order events, group exceptions, prepare customer-service follow-up | Exception queue and approved message drafts | Improve purchase and repeat-purchase workflows |
| Financial services and fintech | Business-customer onboarding operations | Check required-field completeness and route missing-document tasks | Document checklist and status history | Reduce onboarding friction; leave financial decisions to the authorized system and people |
| Insurance operations | Broker inquiries and document follow-up | Route inquiries, check submitted document sets, prepare missing-item notices | Broker worklist and response-time report | Broker acquisition, account servicing and renewal operations |
| Legal and professional services | Inquiry-to-engagement coordination | Deduplicate business inquiries, route service requests, prepare intake checklists | Intake and proposal worklist | Help suitable business inquiries reach the right team; legal advice stays with professionals |
| Recruiting and staffing businesses | Employer-account operations | Clean employer inquiries, route role requirements, track client follow-up | Employer account and service-delivery report | Acquire and retain hiring-company customers; applicant selection is a separate task |
| Education and training providers | Course inquiry and enrollment administration | Group course requests, check administrative completeness, prepare permitted follow-up | Inquiry, registration and completion reports | Help learners discover and adopt offerings; eligibility decisions stay separate |
| Property and real-estate services | Property inquiry and appointment coordination | Route property inquiries, organize viewing requests, check duplicate appointments | Agent worklist and appointment report | Inquiry conversion and broker productivity; no automated housing eligibility decisions |
| Logistics and field services | Shipment or service exception handling | Normalize tracking events, identify missing updates, route service tickets | Exception list with timestamps and owners | Improve customer onboarding and operational retention |
| Hospitality and travel | Booking-request coordination | Check reservation records, route special requests, prepare follow-up drafts | Booking worklist and unresolved requests | Booking conversion, repeat bookings and partner operations |
| Manufacturing and wholesale | Request-for-quote processing | Validate product quantities, route quote requests, assemble supplied catalog information | Quote preparation packet | Distributor acquisition and faster sales handoffs |
| Nonprofits and membership organizations | Member and donor administration | Deduplicate records, check permission fields, organize renewal worklists | Reviewed contact and renewal reports | Recruitment, stewardship and retention |
| Media, events and agencies | Campaign production and registration operations | Prepare content tasks, reconcile event registrations, roll up supplied campaign events | Production checklist and campaign report | Audience growth, sponsor sales and client reporting |

The reusable part is **receive → validate → queue → process → save → review**. The domain-specific part is the data, rules, permissions and definition of a correct result.

## Clinical domains: where the boundary sits

There are two distinct applications:

1. **Clinical operations:** reduce administrative work around appointments, intake, referrals and document coordination.
2. **Healthcare GTM:** help a business sell and deliver products to clinics, hospitals or other healthcare organizations.

Neither requires starting with automated diagnosis. A practical first demonstration is a synthetic appointment worklist: read invented appointment IDs, check missing administrative fields and duplicate time slots, then produce a staff-review report. A worker can crash midway through the batch and another can finish the work.

Real deployment would require the organization's approved data handling, access controls, audit records, retention policy and clinical-system integration. Relay's local database stores input without encryption and is not a clinical deployment package. Removing names alone is not proof that patient information has been de-identified; HHS describes formal methods and requirements in its [de-identification guidance](https://www.hhs.gov/hipaa/for-professionals/special-topics/de-identification/index.html).

Patient reminders, clinical marketing and health-system writes need their own authorized workflows. Appointment urgency, diagnosis, treatment and clinical triage are outside this guide's proposed automation.

## The GTM engineering workbench

This covers the main customer-lifecycle responsibilities. Individual companies combine them differently; it is not a claim that every GTM role owns every row.

| Responsibility | Practical workflow to build | Reviewable output | What still needs integration |
| --- | --- | --- | --- |
| Ideal customer profile and account research | Group supplied company records by declared industry, size and region rules | Segment list with rule explanations | Approved research sources and current company data |
| Lead capture and ingestion | Accept form, event or import records with stable IDs and validation | Accepted records, rejected rows and source history | Website forms, CRM or event platform |
| Data hygiene | Normalize fields, identify duplicate identities and flag conflicting values | Merge proposals; no silent overwrites | Organization-specific matching and merge policy |
| Enrichment | Join supplied records to an approved data source and record provenance | Proposed added fields with source and retrieval date | Authorized provider access and field-use policy |
| Qualification and intent | Apply explicit fit rules and permitted activity signals | Priority suggestions with reasons | Validated rules and real outcome evaluation |
| Ownership and routing | Suggest owners by territory, product and availability | Assignment proposal and unassigned exceptions | CRM ownership rules and team availability |
| Outreach and sales assistance | Prepare drafts from verified account facts and approved messaging | Drafts with sources and suppression checks | Sending provider, contact permissions and human review |
| Demand generation and content operations | Turn a campaign brief into production tasks and approval steps | Content briefs, task status and approved assets | CMS, advertising and publishing tools |
| CRM synchronization and handoffs | Compare incoming data with current records before writing | Change preview, conflicts and handoff checklist | Provider IDs, API limits and explicit write authorization |
| Campaign and funnel measurement | Deduplicate supplied events and calculate declared conversion metrics | Counts, denominators, cohort windows and missing-data warnings | Reliable tracking and agreed event definitions |
| Experiment operations | Validate exposure/conversion records and prepare variant comparisons | Experiment report with uncertainty and limitations | Sound experiment design and actual assignment data |
| Pipeline and revenue operations | Summarize opportunity stages, age and supplied amounts | Pipeline rollup and stale-opportunity worklist | CRM definitions; forecast assumptions must be explicit |
| Product-led growth and activation | Track declared onboarding milestones and route incomplete steps | Activation worklist and cohort report | Product events and customer identity mapping |
| Customer success, renewals and expansion | Identify missing onboarding steps or upcoming renewals from supplied records | Account-review tasks and follow-up drafts | Customer-success systems and human account judgment |
| Partner and channel operations | Reconcile referrals, registrations and partner handoffs | Referral status and conflict report | Partner systems and attribution rules |
| Reliability and cost control | Handle failures, bounded retries, rate limits and replay | Failure dashboard, retry history and cost/volume report | Provider contracts, pricing and production monitoring |

Qualification rules should remain explainable. Fit and engagement are separate signals; Salesforce's [qualification and routing material](https://trailhead.salesforce.com/content/learn/modules/opportunity-management/qualify-and-route-leads-to-your-reps) illustrates that distinction. A homemade score is a business rule, not a proven prediction of conversion.

## One concrete end-to-end example

A software company receives 500 event registrations:

1. Ingest the supplied file and assign stable event/record IDs.
2. Check required fields and flag duplicate or conflicting records.
3. Group company accounts using the company's declared segment rules.
4. Prepare owner assignments and follow-up drafts for review.
5. Export the approved worklist; a later authorized connector can perform CRM writes or sending.
6. Import subsequent activity and report the agreed funnel metrics.

Relay's queue/recovery pattern could carry each batch or record job. A crash should not erase completed work. For external writes or messages, however, a worker may fail after the provider accepted an action but before the local result was saved. Retrying blindly can duplicate the action. A connector needs provider idempotency support or a reconciliation step; a durable local queue alone does not guarantee exactly-once external effects.

The healthcare version changes the input to synthetic appointment records, the rules to administrative completeness and scheduling checks, and the output to a staff worklist. It does not turn lead qualification into patient prioritization.

## What to ship first in each domain

Start with **one input, one deterministic task and one reviewable output**. Demonstrate normal execution, duplicate input, a worker crash and restart. Show the actual result before adding connectors.

| First package | Useful proof | Suitable evidence |
| --- | --- | --- |
| GTM lead-list workroom | Clean and segment supplied prospect data without losing rows | Input/output counts, duplicate/conflict cases, recovery test |
| Synthetic clinic workroom | Produce a complete administrative worklist after a worker crash | Invented input records, expected exceptions, restored report |
| Ecommerce exception workroom | Reconcile supplied order events and route unresolved cases | Duplicate-event tests, exception reasons, replay history |
| Customer onboarding workroom | Track required steps and prepare account follow-up | Missing-step cases, owner proposals, persisted completion |
| Campaign reporting workroom | Reproduce metrics from an explicit event definition | Known counts, denominator tests and source hashes |

A useful portfolio artifact includes a sample dataset, runnable command, visible result, failure demonstration and clear limitations. Production integrations need independent end-to-end tests. Business improvements such as higher conversion, fewer missed appointments or lower churn require measured outcomes, not just a successful queue demonstration.

## Current implementation and next choices

See [Relay's shipped runner](projects/16-reliable-job-runner/README.md), [actual recovery evidence](evidence/job-runner-recovery.json) and [the test workflow](.github/workflows/tests.yml). They establish the existing single-host foundation.

This guide maps applications; it does not ship the proposed domain transformations. Choose a specific domain and workflow before adding its schema, rules, fixtures, tests and connector boundaries. That keeps each next release concrete and reviewable.
