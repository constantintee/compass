# Compass 🧭

A full-stack dockerized stock prediction system leveraging machine learning to forecast market trends.

## Overview

**Compass** is a comprehensive, full-stack application designed to predict stock prices using advanced machine learning algorithms, based on the Elliot-Wave-Theory. The project is fully dockerized, separating concerns into individual containers for streamlined development and deployment. It leverages a PostgreSQL database enhanced with TimescaleDB for efficient time-series data management. The goal is to make stock prdictions good and open-source. I'm not a programmer by any means. I've done all of that with AI (which I'm not so proud about. It's like working with a very smart baby....). I apprichiate any help to keep this project moving forward. 

## Architecture

The application is divided into five main steps:

- **Downloader**: Handles the acquisition of historical stock data
- **Preprocessor**: Cleans and transforms raw data for ML pipeline
- **Training**: Executes machine learning model training
- **Backend**: Manages API endpoints and business logic
- **Frontend**: Provides user interface and data visualization

## Table of Contents

- [GettingStarted](#GettingStarted)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

### Database

The system utilizes PostgreSQL with TimescaleDB extension for efficient time-series data management.

## Technology Stack

- **Programming Language**: Python
- **Web Framework**: Django with HTMX
- **Containerization**: Docker
- **Database**: PostgreSQL + TimescaleDB

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```bash
git clone https://gitlab.com/compass-share/compass.git
cd compass
```

2. Set Up Environment Variables

Each service can be configured through environment variables. Copy the example environment file:
```
cp .env.example .env
```

Edit the .env file with your specific configuration.

3. Each service can be configured through the config.yaml file. Copy the example config file:
```bash
cp config.yaml.example .config.yaml
```

4. Build and start the containers:
```bash
docker-compose up --build
```

## Usage
### Access the Web Application

Open your browser and navigate to http://localhost:8000.

### Explore Features

View stock predictions for selected companies.
Input custom stock symbols to get personalized predictions.
Analyze historical data through interactive charts.


## Contributing
We welcome contributions!

**Fork the Repository**

Click the "Fork" button at the top right corner of the repository page.

**Clone Your Fork**

```bash
git clone https://gitlab.com/yourusername/compass.git
```
**Create a New Branch**

```bash
git checkout -b feature/YourFeature
```

**Make Changes**

Implement your feature or fix.

**Commit Changes**

```bash
git commit -m "Add YourFeature"
```

**Push to Your Branch**

```bash
git push origin feature/YourFeature
```

**Submit a Merge Request**

Go to the original repository and create a merge request from your forked branch.

## License
This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE]([https://gitlab.com/compass-share/compass/-/blob/main/LICENSE?ref_type=heads](https://github.com/constantintee/compass/blob/main/LICENSE)) file for details.
