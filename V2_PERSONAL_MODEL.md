# V2 Personal Memory Model

## Why provenance exists

A personal taste model is not a universal model. It is trained from one person's archive, labels, exclusions, and feature representation.

The model therefore needs a receipt—called **provenance**—that records where it came from and whether it is still valid.

## Provenance example

The following is a provenance example, not the current public default or a claim about another user's archive. A real local model must display its own measured receipt before influencing dispatch.

- the owner's Apple Photos Library;
- Favorite labels;
- Apple Vision feature prints;
- a particular feature-print model version;
- a particular exclusion policy.

Its provenance is conceptually:

```text
Model: personal taste model
Training photos: 83,314
Positive labels: 1,594 Favorites
Embedding model: vision-feature-print-v1
Exclusions applied: none
AUC: 0.853
Status: ready
```

If the owner later excludes a person, that exclusion is a dispatch policy. It immediately filters that person's photos from resurfacing, but it does not automatically invalidate the taste model. This is intentional: the user should be able to make a policy change without waiting for a full rebuild.

A separate, optional clean-model rebuild can remove excluded photos from training and validation. That operation creates a new provenance record and a fresh AUC.

## Exclusion and retraining lifecycle

```text
Hard exclusion saved
        ↓
Exclusion applies immediately to dispatch
        ↓
Existing taste model may continue to rank eligible photos
        ↓
Optional clean-model rebuild requested
        ↓
Excluded photos removed from training and validation
        ↓
Model retrains if enough eligible data exists
        ↓
Fresh AUC is computed
        ↓
Eligible photos are rescored
```

This separates the fast policy action from the slower modeling decision. A hard exclusion is not automatically treated as a negative emotional label.

## AUC lifecycle

AUC is computed when a model is trained or retrained. It is not recomputed on every application startup when the model, eligible dataset, policy, and feature version are unchanged.

A fresh AUC is required when any of these change:

- hard exclusions;
- training photos;
- Favorite labels;
- feedback labels;
- feature-print model version;
- training split or model parameters.

The AUC must always be interpreted with its accompanying provenance:

```text
AUC = 0.853
training photos = 83,314
Favorites = 1,594
exclusions = none
feature model = vision-feature-print-v1
```

An AUC value without that context is not comparable across users or model versions.

## Cold start

A new user does not need to label hundreds of photos before receiving value.

- No personal model: use neutral deterministic resonance.
- Existing Favorites: optionally bootstrap a personal model.
- Too few Favorites: keep the model disabled or strongly shrunk toward neutral.
- Direct feedback: gradually becomes the strongest personal signal.

The personal model is never copied between users.

## What the model is allowed to learn

The model predicts:

> “Would this user want this memory resurfaced?”

It does not decide:

- what an emotion objectively is;
- whether a person is important in the user's life;
- what the historical archive contains;
- whether a photograph is artistically good for everyone.

Historical insights remain generated from the archive and deterministic signals. Personal learning changes attention, not history.
