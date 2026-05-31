# Encountered Issues

## 1. WebShop training crashes when a search action triggers attribute search

### Symptom

During WebShop training, the run can terminate inside a Ray environment worker with an error like:

```text
TypeError: NoneType object is not subscriptable
```

The stack trace points to:

```text
agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py
get_top_n_product_from_keywords
asins = attribute_to_asins[attribute]
```

This is not a model loading issue, batch size issue, or missing WebShop data issue. The crash happens during `env.step(action)` after the model emits a legal `search[...]` action.

### Root Cause

WebShop exposes `search[<your query>]` as an admissible action, but internally its search engine supports hidden special query modes:

```text
<r>  random search
<a>  attribute search
<c>  category search
<q>  query search
plain text  normal keyword search
```

A normal search such as:

```text
search[men athletic shoes rubber sole under 120]
```

uses the Lucene keyword search path and does not need `attribute_to_asins`.

If a sampled model action is parsed as an attribute query, for example:

```text
search[<a> rubber sole]
```

then WebShop uses `attribute_to_asins` to map the attribute to matching product ASINs. The Flask app path passes this mapping, but the Gym `SimServer` path discarded it when loading products and did not pass it into `get_top_n_product_from_keywords`. As a result, `attribute_to_asins` was `None`, and the attribute search branch crashed.

Previous 1.5B runs may have completed because they did not sample a query that entered this hidden `<a>` branch. The bug was latent in the environment code and only appears when one of the parallel rollout workers triggers that branch.

### Why The Whole Training Job Stops

A training step runs many WebShop environments in parallel. For example:

```text
data.train_batch_size=16
env.rollout.n=8
=> 128 parallel trajectories
```

If any one Ray WebShop worker raises an exception inside `env.step(action)`, the parent process receives a `ray.exceptions.RayTaskError` from `ray.get(futures)`. PPO cannot continue without the full batch of observations, rewards, and done flags, so the training job exits.

### Fix Applied

Updated the Gym WebShop server to keep and pass the attribute index returned by `load_products`:

```text
agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py
```

Specifically, `SimServer` now stores `self.attribute_to_asins` and passes it to `get_top_n_product_from_keywords`, matching the Flask app behavior.

Also added a defensive fallback in:

```text
agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/engine.py
```

Unknown or missing attribute search mappings now return an empty result set instead of raising an exception.

### Impact

Normal keyword searches are unchanged. For example, this path behaves as before:

```text
search[men athletic shoes rubber sole under 120]
```

Only hidden attribute search queries are affected:

```text
search[<a> rubber sole]
```

Before the fix, such a query could crash training. After the fix, it performs attribute search if possible, or returns no products if the attribute is unknown.
