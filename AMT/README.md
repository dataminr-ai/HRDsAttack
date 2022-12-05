## Overview
This folder contains all AMT related scripts and files, including:

1. Data processing notebooks for annotation tasks and associated example intermediate files;
2. Data folders ([dataframes](dataframes) and [util_data](util_data)), these are data files using in the data processing notebooks.
3. A HTML hosting environment, including all the annotation UI HTML files.
4. Python scripts related to AMT operations.

### Notebooks:
The notebooks include all steps for data processing, before the annotation task and afterwards. 
* [00_qualification_evaluation.ipynb](notebooks/00_qualification_evaluation.ipynb) contain steps for processing data from qualification tasks, which also generates a report of worker performance based on their submissions.
* [01_annotation_prep.ipynb](notebooks/01_annotation_prep.ipynb) contain steps for preparing annotation batches from raw data.
* [02_post_annotation_process](notebooks/02_post_annotation_process.ipynb) contain steps for post-processing, including data structure conversion, annotation merging, and basic EDA steps.

Note that all these notebooks are for demonstration purposes only. Modify the steps to adapt to your specific use cases.

### Annotation UI Dev App
The [ui_app](ui_app) folder contains a script to launch a development server for hosting different HTML interfaces, including the UI for both qualification task and the full task.

To launch the server, run
```bash
cd AMT/ui_app
python3 app.py
```
Then go to http://127.0.0.1:5050/qualification to access the rendered qualification UI, and http://127.0.0.1:5050/full-task for the rendered full task UI.

The [templates](ui_app/templates) folder contains all HTML files associated to the UIs.

### Notify Workers via AWS CLI
To send notifications to workers, edit the `subject_line`, `message_text`,and `worker_ids` in [notify_workers.py](scripts/notify_workers.py). Then run
```bash
python notify_worker.py
```
If the returned string contains 0 error then the message was successfully sent.