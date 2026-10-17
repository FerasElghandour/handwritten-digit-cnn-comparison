# Release validation

Source syntax and notebook code cells are checked on Python 3.11 and 3.12 by the included workflow. These source checks do not run training, databases, paid model providers, or external simulations. Additional runtime checks are recorded below when executed.

Full historical experiment reproduction is separate from source validation. No new benchmark result is inferred from syntax checks.

## Complete notebook execution

Fresh CPU execution reports baseline test accuracy 97.78% and deeper-model accuracy 99.17%, with macro-F1 0.9777 and 0.9915 on the same 360-row test partition. These fresh results are separate from the saved run above.

All 13 code cells executed successfully in an isolated CPU process. No model-provider calls were made. Source/input hashes and dependency versions are recorded in `results/reproduction.json`. The raw inputs and raw execution logs are not distributed.
