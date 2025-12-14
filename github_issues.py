#!/usr/bin/env python3

import json
import os
from pathlib import Path
from datetime import datetime

class OfflineIssueManager:
    def __init__(self, storage_dir: str = ".github_issues"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.labels_file = self.storage_dir / "labels.json"
        self.milestones_file = self.storage_dir / "milestones.json"
        self.issues_file = self.storage_dir / "issues.json"
        
    def save_labels(self):
        labels = {
            "Infrastructure": "1f50e8",
            "Data-Engineering": "0052cc",
            "Kafka-Streaming": "fbca04",
            "AI-Federated": "d4c5f9",
            "Containerization": "cccccc",
            "Monitoring": "ff6b6b",
            "P1-DevOps": "e11d21",
            "P2-Data": "0e8a16",
            "P3-AI-Client": "c2e0c6",
            "P4-Server": "fef2c0",
            "To-Do": "cccccc",
            "In-Progress": "fbca04",
            "Done": "0e8a16",
            "Blocked": "d73a49"
        }
        
        with open(self.labels_file, "w") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved {len(labels)} labels to {self.labels_file}")
    
    def save_milestones(self):
        milestones = [
            {"title": "Phase 1: Infrastructure", "description": "Infrastructure & Environment Setup"},
            {"title": "Phase 2: Data Pipeline", "description": "Data Engineering (Pipeline de Donnees)"},
            {"title": "Phase 3: Kafka Core", "description": "Kafka Streaming Core"},
            {"title": "Phase 4: AI Logic", "description": "AI & Federated Logic"},
            {"title": "Phase 5: Containerization", "description": "Containerization & Orchestration"},
            {"title": "Phase 6: Monitoring", "description": "Monitoring & Visualization"}
        ]
        
        with open(self.milestones_file, "w") as f:
            json.dump(milestones, f, indent=2)
        print(f"Saved {len(milestones)} milestones to {self.milestones_file}")
    
    def save_issues(self):
        issues = [
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.1 Repository Project Init",
                "body": "Setup initial GitHub repository structure.\n\nTasks:\n- Create project structure\n- Initialize git\n- Add .gitignore\n- Setup repository",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.2 Project Directory Structure",
                "body": "Create project directories.\n\nCreate:\n- client/\n- server/\n- data/{raw,processed}/\n- models/\n- config/\n- tests/",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.3 Virtual Environment Setup",
                "body": "Setup Python virtual environment.\n\nSteps:\n- Create venv\n- Activate\n- Upgrade pip\n- Create requirements.txt",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.4 Dependencies Definition",
                "body": "Define project dependencies.\n\nPackages:\n- apache-spark\n- kafka-python\n- torch\n- flower\n- pandas",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.5 Docker Installation Check",
                "body": "Verify Docker setup.\n\nChecklist:\n- Docker >= 20.10\n- Docker daemon running\n- Docker compose available",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 1: Infrastructure",
                "title": "1.6 Java & Spark Verification",
                "body": "Verify Java and Spark.\n\nRequirements:\n- Java JDK 11+\n- Spark 3.0+\n- JAVA_HOME set\n- SPARK_HOME set",
                "labels": ["Infrastructure", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.1 Client Dockerfile",
                "body": "Create client Dockerfile.\n\nRequirements:\n- Python 3.9+ base\n- Install dependencies\n- Setup working dir\n- Define entry point",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.2 Data Splitting Script",
                "body": "Create data/split_data.py\n\nFunctionality:\n- Load raw data\n- 80/20 split\n- Stratification\n- Save datasets",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.3 Spark Session Optimization",
                "body": "Optimize Spark configuration.\n\nConfig:\n- Memory: 4GB executor\n- Partition optimization\n- Cache strategy\n- Shuffle optimization",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.4 Schema & Cleaning Logic",
                "body": "Create data/schema.py\n\nImplement:\n- StructType schema\n- Null value handling\n- Type validation\n- Feature normalization",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.5 Streaming Ingestion",
                "body": "Setup real-time ingestion.\n\nImplement:\n- Spark Structured Streaming\n- Kafka topic reading\n- Transformations\n- Sink writing",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 2: Data Pipeline",
                "title": "2.6 Data Injector Script",
                "body": "Create data/data_injector.py\n\nFunctionality:\n- Generate synthetic data\n- Send to Kafka\n- Real-time simulation\n- Configurable rate",
                "labels": ["Data-Engineering", "P2-Data", "To-Do"]
            },
            {
                "phase": "Phase 3: Kafka Core",
                "title": "3.1 Docker Compose Kafka Setup",
                "body": "Configure docker-compose.yml for Kafka.\n\nServices:\n- Zookeeper\n- Kafka broker\n- Network config\n- Volume mapping",
                "labels": ["Kafka-Streaming", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 3: Kafka Core",
                "title": "3.2 Kafka Dependencies",
                "body": "Add Kafka packages to requirements.txt\n\nPackages:\n- kafka-python\n- confluent-kafka\n- pyspark Kafka connector",
                "labels": ["Kafka-Streaming", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 3: Kafka Core",
                "title": "3.3 Topic Initialization Script",
                "body": "Create config/kafka_init.py\n\nFunctionality:\n- Create topics\n- Set partitions\n- Replication factor\n- Retention policies",
                "labels": ["Kafka-Streaming", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 3: Kafka Core",
                "title": "3.4 Serialization Utilities",
                "body": "Create config/serialization.py\n\nImplement:\n- Pickle serialization\n- Custom encoder/decoder\n- Compression handling",
                "labels": ["Kafka-Streaming", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 3: Kafka Core",
                "title": "3.5 Communication Test (Ping-Pong)",
                "body": "Create tests/test_kafka_comm.py\n\nTest:\n- Producer sends ping\n- Consumer receives\n- Round-trip verification",
                "labels": ["Kafka-Streaming", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.1 Neural Network Architecture",
                "body": "Create models/heartnet.py\n\nImplement:\n- PyTorch HeartNet\n- ECG signal input\n- Classification head\n- Federated support",
                "labels": ["AI-Federated", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.2 Local Trainer Function",
                "body": "Create models/trainer.py\n\nFunctionality:\n- Train on local data\n- Batch training\n- Gradient computation\n- Model checkpoints",
                "labels": ["AI-Federated", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.3 Weight Serialization",
                "body": "Create models/serializer.py\n\nImplement:\n- Pickle-based storage\n- Version management\n- Checkpointing",
                "labels": ["AI-Federated", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.4 Flower Integration",
                "body": "Integrate Flower framework.\n\nImplement:\n- Custom FedAvg strategy\n- Aggregation logic\n- Client-server comm",
                "labels": ["AI-Federated", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.5 Kafka-Client Logic",
                "body": "Create client/kafka_consumer.py\n\nFunctionality:\n- Consume model updates\n- Apply to local model\n- Send updates back",
                "labels": ["AI-Federated", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 4: AI Logic",
                "title": "4.6 Kafka-Server Logic",
                "body": "Create server/kafka_aggregator.py\n\nFunctionality:\n- Aggregate updates\n- Broadcast weights\n- Manage rounds",
                "labels": ["AI-Federated", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.1 Server Dockerfile",
                "body": "Create server/Dockerfile\n\nRequirements:\n- Lightweight Python image\n- Server dependencies\n- Port exposure\n- Health check",
                "labels": ["Containerization", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.2 Client Dockerfile Hybrid",
                "body": "Create client/Dockerfile\n\nRequirements:\n- Spark dependencies\n- PyTorch support\n- Training capabilities\n- Size optimization",
                "labels": ["Containerization", "P3-AI-Client", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.3 Global Docker Compose",
                "body": "Complete docker-compose.yml\n\nInclude:\n- All services\n- Network config\n- Dependencies\n- Env variables",
                "labels": ["Containerization", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.4 Kafka Listeners Config",
                "body": "Configure Kafka listeners.\n\nSetup:\n- PLAINTEXT listeners\n- Advertised hosts\n- Port bindings\n- Authentication",
                "labels": ["Containerization", "P4-Server", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.5 Volume Management",
                "body": "Setup persistent volumes.\n\nVolumes:\n- Kafka data\n- Database\n- Model checkpoints\n- Logs",
                "labels": ["Containerization", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 5: Containerization",
                "title": "5.6 Integration Tests",
                "body": "Create tests/test_integration.py\n\nTest:\n- Service startup\n- Inter-service comm\n- End-to-end workflow\n- Cleanup",
                "labels": ["Containerization", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.1 Prometheus Client",
                "body": "Integrate Prometheus metrics.\n\nImplement:\n- prometheus_client\n- Custom metrics\n- Metrics endpoint\n- Training metrics",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.2 Prometheus Configuration",
                "body": "Create config/prometheus.yml\n\nSetup:\n- Scrape targets\n- Intervals\n- Service discovery\n- Retention",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.3 Grafana Configuration",
                "body": "Setup Grafana provisioning.\n\nImplement:\n- Data sources\n- Dashboard provisioning\n- Alert rules\n- Notifications",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.4 Monitoring Stack",
                "body": "Add monitoring to docker-compose.yml\n\nAdd:\n- Prometheus container\n- Grafana container\n- Network config\n- Volumes",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.5 Grafana Dashboards",
                "body": "Create Grafana dashboards.\n\nDashboards:\n- Training metrics\n- System resources\n- Kafka metrics\n- Model performance",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            },
            {
                "phase": "Phase 6: Monitoring",
                "title": "6.6 Final Testing & Documentation",
                "body": "Complete testing and documentation.\n\nDeliverables:\n- Test coverage\n- System docs\n- Deployment guide\n- API docs",
                "labels": ["Monitoring", "P1-DevOps", "To-Do"]
            }
        ]
        
        with open(self.issues_file, "w") as f:
            json.dump(issues, f, indent=2)
        print(f"Saved {len(issues)} issues to {self.issues_file}")
    
    def generate_github_script(self):
        script_content = '''#!/usr/bin/env python3

import json
import subprocess
import sys
from pathlib import Path

storage_dir = Path(".github_issues")

with open(storage_dir / "labels.json") as f:
    labels = json.load(f)

with open(storage_dir / "milestones.json") as f:
    milestones = json.load(f)

with open(storage_dir / "issues.json") as f:
    issues = json.load(f)

print("Creating labels...")
for label, color in labels.items():
    cmd = ["gh", "label", "create", label, "--color", color, "--force"]
    subprocess.run(cmd, capture_output=True)

print("Creating milestones...")
for ms in milestones:
    cmd = ["gh", "milestone", "create", ms["title"], "--description", ms["description"]]
    subprocess.run(cmd, capture_output=True)

print("Creating issues...")
for issue in issues:
    cmd = [
        "gh", "issue", "create",
        "--title", issue["title"],
        "--body", issue["body"]
    ]
    for label in issue["labels"]:
        cmd.extend(["--label", label])
    
    phase = issue["phase"]
    cmd.extend(["--milestone", phase])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK] {issue['title']}")
    else:
        print(f"  [SKIP] {issue['title']}")

print(f"\\nTotal issues configured: {len(issues)}")
print("Run this script when you have internet connection:")
print("  python upload_to_github.py")
'''
        
        with open("upload_to_github.py", "w") as f:
            f.write(script_content)
        print("Created: upload_to_github.py (for when you have internet)")
    
    def run(self):
        print("\nCardiologie Project - Offline Issues Manager")
        print("=" * 50)
        print("")
        
        self.save_labels()
        self.save_milestones()
        self.save_issues()
        self.generate_github_script()
        
        print("")
        print("=" * 50)
        print("Offline storage completed!")
        print("=" * 50)
        print("")
        print("Files created:")
        print(f"  - {self.labels_file}")
        print(f"  - {self.milestones_file}")
        print(f"  - {self.issues_file}")
        print(f"  - upload_to_github.py")
        print("")
        print("When you have internet again, run:")
        print("  python upload_to_github.py")
        print("")
        print("This will push all issues to GitHub automatically!")


def main():
    manager = OfflineIssueManager()
    manager.run()


if __name__ == "__main__":
    main()