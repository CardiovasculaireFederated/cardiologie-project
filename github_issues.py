#!/usr/bin/env python3

import subprocess
import sys
from typing import List

class GitHubIssueCreator:
    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_full = f"{repo_owner}/{repo_name}"
        
    def create_labels(self):
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
        
        print("\nCreating labels...")
        for label, color in labels.items():
            cmd = [
                "gh", "label", "create", label,
                "--repo", self.repo_full,
                "--color", color,
                "--force"
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  [OK] Label '{label}'")
            except subprocess.CalledProcessError:
                print(f"  [EXISTS] Label '{label}'")
    
    def create_milestone(self, title: str, description: str):
        cmd = [
            "gh", "milestone", "create", title,
            "--repo", self.repo_full,
            "--description", description
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  [OK] Milestone '{title}'")
        except subprocess.CalledProcessError:
            print(f"  [EXISTS] Milestone '{title}'")
    
    def create_milestones(self):
        print("\nCreating milestones...")
        milestones = [
            ("Phase 1: Infrastructure", "Infrastructure & Environment Setup"),
            ("Phase 2: Data Pipeline", "Data Engineering (Pipeline de Donnees)"),
            ("Phase 3: Kafka Core", "Kafka Streaming Core"),
            ("Phase 4: AI Logic", "AI & Federated Logic"),
            ("Phase 5: Containerization", "Containerization & Orchestration"),
            ("Phase 6: Monitoring", "Monitoring & Visualization")
        ]
        for title, desc in milestones:
            self.create_milestone(title, desc)
    
    def create_issue(self, title: str, body: str, labels: List[str], milestone: str = None) -> bool:
        cmd = [
            "gh", "issue", "create",
            "--repo", self.repo_full,
            "--title", title,
            "--body", body
        ]
        
        for label in labels:
            cmd.extend(["--label", label])
        
        if milestone:
            cmd.extend(["--milestone", milestone])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def create_all_issues(self):
        print("\nCreating GitHub Issues...\n")
        
        issues = [
            ("Phase 1: Infrastructure", [
                ("1.1 Repository Project Init", 
                 "Setup initial GitHub repository structure.\n\nTasks:\n- Create project structure\n- Initialize git\n- Add .gitignore\n- Setup repository",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
                ("1.2 Project Directory Structure",
                 "Create project directories.\n\nCreate:\n- client/\n- server/\n- data/{raw,processed}/\n- models/\n- config/\n- tests/",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
                ("1.3 Virtual Environment Setup",
                 "Setup Python virtual environment.\n\nSteps:\n- Create venv\n- Activate\n- Upgrade pip\n- Create requirements.txt",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
                ("1.4 Dependencies Definition",
                 "Define project dependencies.\n\nPackages:\n- apache-spark\n- kafka-python\n- torch\n- flower\n- pandas",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
                ("1.5 Docker Installation Check",
                 "Verify Docker setup.\n\nChecklist:\n- Docker >= 20.10\n- Docker daemon running\n- Docker compose available",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
                ("1.6 Java & Spark Verification",
                 "Verify Java and Spark.\n\nRequirements:\n- Java JDK 11+\n- Spark 3.0+\n- JAVA_HOME set\n- SPARK_HOME set",
                 ["Infrastructure", "P1-DevOps", "To-Do"]),
            ]),
            ("Phase 2: Data Pipeline", [
                ("2.1 Client Dockerfile",
                 "Create client Dockerfile.\n\nRequirements:\n- Python 3.9+ base\n- Install dependencies\n- Setup working dir\n- Define entry point",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
                ("2.2 Data Splitting Script",
                 "Create data/split_data.py\n\nFunctionality:\n- Load raw data\n- 80/20 split\n- Stratification\n- Save datasets",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
                ("2.3 Spark Session Optimization",
                 "Optimize Spark configuration.\n\nConfig:\n- Memory: 4GB executor\n- Partition optimization\n- Cache strategy\n- Shuffle optimization",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
                ("2.4 Schema & Cleaning Logic",
                 "Create data/schema.py\n\nImplement:\n- StructType schema\n- Null value handling\n- Type validation\n- Feature normalization",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
                ("2.5 Streaming Ingestion",
                 "Setup real-time ingestion.\n\nImplement:\n- Spark Structured Streaming\n- Kafka topic reading\n- Transformations\n- Sink writing",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
                ("2.6 Data Injector Script",
                 "Create data/data_injector.py\n\nFunctionality:\n- Generate synthetic data\n- Send to Kafka\n- Real-time simulation\n- Configurable rate",
                 ["Data-Engineering", "P2-Data", "To-Do"]),
            ]),
            ("Phase 3: Kafka Core", [
                ("3.1 Docker Compose Kafka Setup",
                 "Configure docker-compose.yml for Kafka.\n\nServices:\n- Zookeeper\n- Kafka broker\n- Network config\n- Volume mapping",
                 ["Kafka-Streaming", "P4-Server", "To-Do"]),
                ("3.2 Kafka Dependencies",
                 "Add Kafka packages to requirements.txt\n\nPackages:\n- kafka-python\n- confluent-kafka\n- pyspark Kafka connector",
                 ["Kafka-Streaming", "P4-Server", "To-Do"]),
                ("3.3 Topic Initialization Script",
                 "Create config/kafka_init.py\n\nFunctionality:\n- Create topics\n- Set partitions\n- Replication factor\n- Retention policies",
                 ["Kafka-Streaming", "P4-Server", "To-Do"]),
                ("3.4 Serialization Utilities",
                 "Create config/serialization.py\n\nImplement:\n- Pickle serialization\n- Custom encoder/decoder\n- Compression handling",
                 ["Kafka-Streaming", "P4-Server", "To-Do"]),
                ("3.5 Communication Test (Ping-Pong)",
                 "Create tests/test_kafka_comm.py\n\nTest:\n- Producer sends ping\n- Consumer receives\n- Round-trip verification",
                 ["Kafka-Streaming", "P4-Server", "To-Do"]),
            ]),
            ("Phase 4: AI Logic", [
                ("4.1 Neural Network Architecture",
                 "Create models/heartnet.py\n\nImplement:\n- PyTorch HeartNet\n- ECG signal input\n- Classification head\n- Federated support",
                 ["AI-Federated", "P3-AI-Client", "To-Do"]),
                ("4.2 Local Trainer Function",
                 "Create models/trainer.py\n\nFunctionality:\n- Train on local data\n- Batch training\n- Gradient computation\n- Model checkpoints",
                 ["AI-Federated", "P3-AI-Client", "To-Do"]),
                ("4.3 Weight Serialization",
                 "Create models/serializer.py\n\nImplement:\n- Pickle-based storage\n- Version management\n- Checkpointing",
                 ["AI-Federated", "P3-AI-Client", "To-Do"]),
                ("4.4 Flower Integration",
                 "Integrate Flower framework.\n\nImplement:\n- Custom FedAvg strategy\n- Aggregation logic\n- Client-server comm",
                 ["AI-Federated", "P3-AI-Client", "To-Do"]),
                ("4.5 Kafka-Client Logic",
                 "Create client/kafka_consumer.py\n\nFunctionality:\n- Consume model updates\n- Apply to local model\n- Send updates back",
                 ["AI-Federated", "P3-AI-Client", "To-Do"]),
                ("4.6 Kafka-Server Logic",
                 "Create server/kafka_aggregator.py\n\nFunctionality:\n- Aggregate updates\n- Broadcast weights\n- Manage rounds",
                 ["AI-Federated", "P4-Server", "To-Do"]),
            ]),
            ("Phase 5: Containerization", [
                ("5.1 Server Dockerfile",
                 "Create server/Dockerfile\n\nRequirements:\n- Lightweight Python image\n- Server dependencies\n- Port exposure\n- Health check",
                 ["Containerization", "P4-Server", "To-Do"]),
                ("5.2 Client Dockerfile Hybrid",
                 "Create client/Dockerfile\n\nRequirements:\n- Spark dependencies\n- PyTorch support\n- Training capabilities\n- Size optimization",
                 ["Containerization", "P3-AI-Client", "To-Do"]),
                ("5.3 Global Docker Compose",
                 "Complete docker-compose.yml\n\nInclude:\n- All services\n- Network config\n- Dependencies\n- Env variables",
                 ["Containerization", "P1-DevOps", "To-Do"]),
                ("5.4 Kafka Listeners Config",
                 "Configure Kafka listeners.\n\nSetup:\n- PLAINTEXT listeners\n- Advertised hosts\n- Port bindings\n- Authentication",
                 ["Containerization", "P4-Server", "To-Do"]),
                ("5.5 Volume Management",
                 "Setup persistent volumes.\n\nVolumes:\n- Kafka data\n- Database\n- Model checkpoints\n- Logs",
                 ["Containerization", "P1-DevOps", "To-Do"]),
                ("5.6 Integration Tests",
                 "Create tests/test_integration.py\n\nTest:\n- Service startup\n- Inter-service comm\n- End-to-end workflow\n- Cleanup",
                 ["Containerization", "P1-DevOps", "To-Do"]),
            ]),
            ("Phase 6: Monitoring", [
                ("6.1 Prometheus Client",
                 "Integrate Prometheus metrics.\n\nImplement:\n- prometheus_client\n- Custom metrics\n- Metrics endpoint\n- Training metrics",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
                ("6.2 Prometheus Configuration",
                 "Create config/prometheus.yml\n\nSetup:\n- Scrape targets\n- Intervals\n- Service discovery\n- Retention",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
                ("6.3 Grafana Configuration",
                 "Setup Grafana provisioning.\n\nImplement:\n- Data sources\n- Dashboard provisioning\n- Alert rules\n- Notifications",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
                ("6.4 Monitoring Stack",
                 "Add monitoring to docker-compose.yml\n\nAdd:\n- Prometheus container\n- Grafana container\n- Network config\n- Volumes",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
                ("6.5 Grafana Dashboards",
                 "Create Grafana dashboards.\n\nDashboards:\n- Training metrics\n- System resources\n- Kafka metrics\n- Model performance",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
                ("6.6 Final Testing & Documentation",
                 "Complete testing and documentation.\n\nDeliverables:\n- Test coverage\n- System docs\n- Deployment guide\n- API docs",
                 ["Monitoring", "P1-DevOps", "To-Do"]),
            ]),
        ]
        
        total_issues = 0
        created_issues = 0
        
        for phase, phase_issues in issues:
            print(f"\n{phase}")
            print("-" * 50)
            
            for title, body, labels in phase_issues:
                total_issues += 1
                milestone = phase.split(":")[0]
                success = self.create_issue(title, body, labels, milestone)
                
                if success:
                    print(f"  [OK] {title}")
                    created_issues += 1
                else:
                    print(f"  [SKIP] {title} (might exist)")
        
        print("\n" + "="*50)
        print(f"Issues: {created_issues}/{total_issues} created")
        print("="*50)
    
    def run(self):
        print("\nCardiologie Project - GitHub Issues Creator")
        print("="*50)
        
        self.create_labels()
        self.create_milestones()
        self.create_all_issues()
        
        print("\nGitHub automation completed!")
        print(f"Repository: {self.repo_full}")


def main():
    repo_owner = "CardiovasculaireFederated"
    repo_name = "cardiologie-project"
    
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Error: GitHub CLI not authenticated")
            print("Run: gh auth login")
            sys.exit(1)
    except FileNotFoundError:
        print("Error: GitHub CLI not installed")
        sys.exit(1)
    
    creator = GitHubIssueCreator(repo_owner, repo_name)
    creator.run()


if __name__ == "__main__":
    main()