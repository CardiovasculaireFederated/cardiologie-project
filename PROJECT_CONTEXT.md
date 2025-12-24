\# PROJECT CONTEXT \& ARCHITECTURE BIBLE

\# Project: Cardio Federated Learning (Medical AI)

\# Status: Sprint 2 - Building Client Architecture



---



\## 1. Project Vision

We are building a \*\*Federated Learning System\*\* for cardiovascular disease prediction.

\*\*Core Philosophy:\*\*

\- \*\*Privacy First:\*\* Patient data NEVER leaves the hospital (Client). Only model weights are shared.

\- \*\*Robustness:\*\* The system must handle corrupt data and connection drops gracefully.

\- \*\*Explainability (XAI):\*\* Medical decisions must be explainable (SHAP/LIME).

\- \*\*Differential Privacy:\*\* Noise is added to weights before transmission.



---



\## 2. System Architecture (The "New" Flow)

The AI MUST follow this exact flow for the Client Logic. Do not deviate.



```mermaid

graph TD

&nbsp;   A\[Kafka: Global Model Arrives] -->|Payload| B(Manager Agent)

&nbsp;   

&nbsp;   subgraph "Hospital Client Environment"

&nbsp;       B --> |1. Assign Task| C{Data Validator Agent}

&nbsp;       C --> |Data Corrupt| Z\[Log Error \& Skip]

&nbsp;       C --> |Data Clean| D{Scout Agent}

&nbsp;       

&nbsp;       D --> |Hyperparameter Search| E\[Quick Test]

&nbsp;       E --> |Optimal Params| B

&nbsp;       

&nbsp;       B --> |2. Initiate Training| F\[Training Agent]

&nbsp;       F --> |Train Epoch| G\[PyTorch Model]

&nbsp;       

&nbsp;       subgraph "Supervision Cycle"

&nbsp;           G -.-> |Metrics: Loss/Acc| H{Sentinel Agent}

&nbsp;           H --> |Stable/Improving| F

&nbsp;           H --> |Early Stopping/Converged| I\[Model Evaluator]

&nbsp;       end

&nbsp;       

&nbsp;       I --> |3. Explainability Check| J\[XAI Agent]

&nbsp;       J --> |SHAP Values ok| K\[Privacy Guard]

&nbsp;       K --> |Apply DP Noise| L\[Encryption Module]

&nbsp;   end

&nbsp;   

&nbsp;   L --> |Serialized Update| M\[Kafka: Send Update]









3\. Directory Structure \& Key Files

client/: Contains all agent logic.



client.py: Entry point. Contains the ManagerAgent.



preprocessing.py: FINISHED. Contains Spark schemas and feature definitions.



trainer.py: TODO. Contains PyTorch training logic.



scout.py: TODO. Hyperparameter optimization.



validator.py: TODO. Data quality checks.



server/: Central aggregation logic.



common/: Shared utilities (Kafka topics, Serialization).



data/: Local datasets (train.csv, test.csv).



4\. Tech Stack \& Tools



ML Framework: PyTorch



Data Processing: PySpark (for preprocessing), Pandas (for local manipulation)



Communication: Apache Kafka (using kafka-python)



Containerization: Docker \& Docker Compose



5\. Coding Rules (The "Vibe")

Strict Type Hinting: Use def func(x: int) -> str: everywhere.



Medical Grade Reliability: No silent failures. Use try-except blocks with logging.



Modular Agents: Each agent (Scout, Trainer, Sentinel) must be a separate class/file.



Existing Code: Respect preprocessing.py. Do not rewrite schemas defined there.



Language: Code comments in English. Variable names in English.



6\. Current Task (Sprint 2)

Goal: Implement the ManagerAgent and the ScoutAgent.



Step-by-Step Requirements:



ManagerAgent (in client.py) listens to global\_model\_updates topic.



Upon receiving a model, it calls preprocessing.py to validate schema.



If valid, it triggers ScoutAgent.



ScoutAgent runs a quick training loop (few epochs) on a subset to find the best Learning Rate.



Report back to Manager.



\# 🚨 PROJECT GUARDRails – READ FIRST 🚨



This file defines the \*\*strict boundaries\*\* of the project.

Any code, suggestion, or modification MUST respect this document.

If something is unclear, ASK — do not assume.



---



\## 🧠 Project Overview



This project is a \*\*Real-Time Federated Learning System for Healthcare (Cardiology)\*\*.



Key characteristics:

\- Event-driven (Kafka)

\- Federated Learning (client-side training)

\- Real-time (system never stops)

\- Scalable (multiple hospitals can join anytime)

\- Privacy-preserving (no raw data leaves hospitals)

\- Production-oriented (not a toy project)



---



\## 🧩 High-Level Architecture



\### Components

\- \*\*Client (Hospital)\*\*:

&nbsp; - Receives global model via Kafka

&nbsp; - Validates local data

&nbsp; - Chooses hyperparameters dynamically

&nbsp; - Trains model locally

&nbsp; - Sends model updates + metadata



\- \*\*Kafka\*\*:

&nbsp; - Message broker

&nbsp; - Topics:

&nbsp;   - `global\_model`

&nbsp;   - `client\_weights`



\- \*\*Server (Federated Coordinator)\*\*:

&nbsp; - Collects client updates

&nbsp; - Aggregates weights

&nbsp; - Sends updated global model

&nbsp; - DOES NOT access raw data



---



\## 🔁 Federated Round Definition



A "Round" means:

1\. Global model is published

2\. Clients may participate (or skip)

3\. Each client decides locally:

&nbsp;  - Data quality

&nbsp;  - Hyperparameters

4\. Clients train and send updates

5\. Server aggregates and publishes new model



⚠️ The system must support:

\- 1 client only

\- N clients

\- Clients joining/leaving at any time



---



\## 🧪 Sprint Status (DO NOT BREAK)



\### ✅ Sprint 1 (DONE)

\- Project structure

\- Docker / Kafka setup

\- Basic client/server skeleton

