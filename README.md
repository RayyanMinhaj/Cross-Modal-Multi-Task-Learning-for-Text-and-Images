# Cross-Modal Multi-Task Learning for Event Extraction

This repository implements a comprehensive pipeline for **event extraction from both text and images** using multi-task learning. The project is structured around four sequential tasks (T1-T4) that progressively extract event information from multimodal data.

## Overview

The system performs event extraction using a modular approach with four interconnected tasks:

1. **T1 - Trigger Detection (Text)**: Identifies event trigger words in sentences using BIO tagging
2. **T2 - Argument Role Classification (Text)**: Classifies the semantic roles of event arguments
3. **T3 - Event Type Classification (Image)**: Predicts event types from images using vision-language models
4. **T4 - Argument Role Extraction (Image)**: Extracts and classifies visual arguments based on bounding boxes

## Project Structure

```
Cross-Modal-Multi-Task-Learning/
├── predict_pipeline.ipynb           
├── t1_trigger_detection_model.py   
├── t2_argument_extraction_model.py  
├── t3_verb_classification.py        
├── t4_role_classification.py        
├── data/                            
│   ├── sample_text_input.json       
│   ├── sample_img_input.json        
│   ├── sample_imgs/                 
│   ├── m2e2/                        

├── models/                          
│   ├── t1_trigger_detection_model.pt
│   ├── t2_argument_extraction_model.pt
│   ├── t3_verb_classification_model.pt
│   └── t4_argument_extraction_model.pt
└── eval/                            
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.13+
- Transformers library
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Cross-Modal-Multi-Task-Learning

# Install required packages
pip install torch transformers pillow torchcrf scikit-learn tqdm
```

## Quick Start

### Running the Pipeline

Train each model in their respective python files, and then open and run the `predict_pipeline.ipynb` notebook to execute the full event extraction pipeline:

```python
# Set your input IDs
sentence_id = "VOA_EN_NW_2017.02.03.3705362_2"
image_id = "VOA_EN_NW_2016.05.11.3325807_1"

# The notebook will:
# 1. Load the corresponding text and image data
# 2. Extract trigger words and events from text
# 3. Classify argument roles from text
# 4. Classify event types from images
# 5. Extract and classify visual arguments
```

### Available Sample Data

**Text IDs:**
- VOA_EN_NW_2017.02.02.3702962_1
- VOA_EN_NW_2017.02.03.3705362_2
- VOA_EN_NW_2016.06.27.3393616_20
- VOA_EN_NW_2016.05.12.3327048_8
- VOA_EN_NW_2016.10.13.3549797_27

**Image IDs:**
- VOA_EN_NW_2016.05.11.3325807_1
- VOA_EN_NW_2016.04.19.3291980_3
- VOA_EN_NW_2015.05.14.2767071_0
- VOA_EN_NW_2014.12.17.2562583_0
- VOA_EN_NW_2016.10.19.3557471_0

## Task Details

### T1: Trigger Detection
- **Model**: BERT + CRF
- **Task**: Named entity recognition for event triggers
- **Input**: Tokenized sentences
- **Output**: BIO labels (B-TRIGGER, I-TRIGGER, O)
- **Tags**: Beginning, Inside, Outside annotations

### T2: Argument Role Classification
- **Model**: BERT classifier
- **Task**: Multi-class classification of argument roles
- **Input**: Prompted sentences with argument markers ($ARG$ ... $/ARG$)
- **Output**: Argument role labels
- **Roles**: Agent, Artifact, Attacker, Destination, Entity, Giver, Instrument, Origin, Person, Place, Police, Recipient, Target, Vehicle, Victim

### T3: Event Type Classification
- **Model**: CLIP Vision-Language Model
- **Task**: Image-based event classification
- **Input**: Image pixel values
- **Output**: Event type labels
- **Event Types**: Life:Die, Movement:Transport, Transaction:Transfer-Money, Conflict:Attack, Conflict:Demonstrate, Contact:Meet, Contact:Phone-Write, Justice:Arrest-Jail

### T4: Argument Role Extraction
- **Model**: CLIP-based classifier on cropped regions
- **Task**: Classify visual arguments based on bounding box regions
- **Input**: Image crops corresponding to annotated regions
- **Output**: Argument role labels (same as T2)

## Data Format

### Text Input (`sample_text_input.json`)
```json
{
  "sentence_id": "VOA_EN_NW_2017.02.02.3702962_1",
  "sentence": "The Post report said Trump had described...",
  "golden-event-mentions": [
    {
      "trigger": {
        "start": 8,
        "end": 9,
        "text": "call"
      },
      "event_type": "Contact:Phone-Write",
      "arguments": [
        {
          "role": "Entity",
          "start": 4,
          "end": 5,
          "text": "Trump"
        }
      ]
    }
  ]
}
```

### Image Input (`sample_img_input.json`)
```json
{
  "VOA_EN_NW_2016.05.11.3325807_1": {
    "role": {
      "Entity": [[region_id, x1, y1, x2, y2], ...],
      "Police": [[region_id, x1, y1, x2, y2], ...]
    },
    "event_type": "Contact:Meet"
  }
}
```

## Model Architectures

### T1 & T2: BERT-based Models
- Base Model: BERT-base-uncased
- T1 adds CRF layer for sequence labeling
- T2 uses pooled output for classification

### T3 & T4: CLIP-based Models
- Base Model: OpenAI CLIP-ViT-Base-Patch32
- Uses visual encoder for image understanding
- Processes full images (T3) or cropped regions (T4)

## Usage Examples

### Training a Task

```python
python t4_role_classification.py
```



## Datasets

This project supports multiple event extraction datasets:

- **M2E2**: Multimodal Event Extraction dataset with images, text, and structured annotations
- **ACE2005**: Automatic Content Extraction dataset with event annotations
- **SWiG/ImSitu**: Situation recognition dataset focusing on visual grounding

## Evaluation

Metrics used for evaluation:
- **Trigger Detection (T1)**: Precision, Recall, F1-score
- **Argument Classification (T2)**: Accuracy, Precision, Recall, F1-score
- **Event Classification (T3)**: Accuracy, Precision, Recall, F1-score
- **Visual Argument Classification (T4)**: Accuracy, Precision, Recall, F1-score


## License

This project is provided as-is for research and educational purposes.
