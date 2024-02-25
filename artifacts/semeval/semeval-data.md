# SemEval Data

https://github.com/ai-systems/Task-2-SemEval-2024/blob/main/training_data.zip

* **train.json** file provides the Premise-Statement-Evidence information
* **dev.json** ???
* **practice_test.json** contains the practice test set, predictions for this set should be submitted to the practice task
* **test.json** contains the test set, predictions for this set should be submitted to the Semeval 2024 task 2. Coming the 10th January 2024!
* CT json folder contains the full set of complete CTRs in individual json files

# Submission
The script takes one single prediction file as input, which MUST be a compressed .json file, named "results.json".

The script takes one single prediction file as input, which MUST be a .json file, named "results.json", structured as follows:

```json
{

    "5bc844fc-e852-4270-bfaf-36ea9eface3d": {

        "Prediction": "Contradiction"

    },

    "86b7cb3d-6186-4a04-9aa6-b174ab764eed": {

        "Prediction": "Contradiction"

    },
}
```