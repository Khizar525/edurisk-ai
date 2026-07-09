"""
Shared test fixtures for EduRisk AI.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame mimicking the dataset structure."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(17, 30, n),
        "Academic Pressure": np.random.randint(1, 6, n),
        "CGPA": np.round(np.random.uniform(1.0, 4.0, n), 2),
        "Study Satisfaction": np.random.randint(1, 6, n),
        "Sleep Duration": np.random.choice(
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], n
        ),
        "Dietary Habits": np.random.choice(["Healthy", "Moderate", "Unhealthy"], n),
        "Work/Study Hours": np.random.randint(0, 13, n),
        "Financial Stress": np.random.randint(1, 6, n),
        "Family History of Mental Illness": np.random.choice(["Yes", "No"], n),
        "Have you ever had suicidal thoughts ?": np.random.choice(["Yes", "No"], n),
        "Depression": np.random.choice([0, 1], n),
    })


@pytest.fixture
def sample_features():
    """Create sample feature array."""
    np.random.seed(42)
    return np.random.randn(100, 11)


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    np.random.seed(42)
    return np.random.choice([0, 1, 2], 100, p=[0.5, 0.3, 0.2])
