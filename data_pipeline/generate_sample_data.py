"""Generate small Synthea-style synthetic CSV files for MediCheck.

The generated rows are synthetic software-test fixtures for MediCheck demos.
They are not intended to imitate real patients or support clinical decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_N_PATIENTS = 1000
DEFAULT_SEED = 42
REFERENCE_DATE = pd.Timestamp("2026-01-01")


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return a logistic transform for vectorized risk simulation."""
    return 1.0 / (1.0 + np.exp(-values))


def build_patient_profile(n_patients: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Create patient demographics and latent disease flags used by all tables."""
    rng = np.random.default_rng(seed)
    patient_ids = [f"P{idx:06d}" for idx in range(1, n_patients + 1)]
    ages = rng.integers(25, 90, size=n_patients)
    bmi = np.clip(rng.normal(loc=27.5, scale=5.3, size=n_patients), 16.0, 48.0)

    diabetes_prob = sigmoid(-6.2 + 0.055 * ages + 0.18 * (bmi - 25.0))
    hypertension_prob = sigmoid(-5.8 + 0.065 * ages + 0.11 * (bmi - 25.0))
    obesity_flag = (bmi >= 30.0).astype(int)
    diabetes_flag = (rng.random(n_patients) < diabetes_prob).astype(int)
    hypertension_flag = (rng.random(n_patients) < hypertension_prob).astype(int)
    cvd_prob = sigmoid(
        -8.4
        + 0.075 * ages
        + 1.1 * diabetes_flag
        + 1.0 * hypertension_flag
        + 0.7 * obesity_flag
        + 0.04 * (bmi - 25.0)
    )
    cvd_flag = (rng.random(n_patients) < cvd_prob).astype(int)

    if n_patients >= 1:
        diabetes_flag[0] = 1
    if n_patients >= 2:
        hypertension_flag[1] = 1
    if n_patients >= 3:
        obesity_flag[2] = 1
        bmi[2] = max(bmi[2], 32.0)
    if n_patients >= 4:
        cvd_flag[3] = 1

    birthdates = [
        (REFERENCE_DATE - pd.to_timedelta(int(age * 365.25), unit="D")).date().isoformat()
        for age in ages
    ]

    return pd.DataFrame(
        {
            "Id": patient_ids,
            "BIRTHDATE": birthdates,
            "SEX": rng.choice(["M", "F"], size=n_patients, p=[0.49, 0.51]),
            "sim_age": ages,
            "sim_bmi": bmi,
            "sim_diabetes": diabetes_flag,
            "sim_hypertension": hypertension_flag,
            "sim_obesity": obesity_flag,
            "sim_cvd": cvd_flag,
        }
    )


def make_patients(profile: pd.DataFrame) -> pd.DataFrame:
    """Return the public patients.csv table."""
    return profile[["Id", "BIRTHDATE", "SEX"]].copy()


def make_encounters(profile: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create Synthea-style encounter rows."""
    rows: list[dict[str, object]] = []
    for patient in profile.itertuples(index=False):
        chronic_burden = int(patient.sim_diabetes) + int(patient.sim_hypertension) + int(patient.sim_obesity) + int(patient.sim_cvd)
        visit_count = max(1, int(rng.poisson(2 + chronic_burden)))
        for visit_idx in range(visit_count):
            start = REFERENCE_DATE - pd.to_timedelta(int(rng.integers(5, 900)), unit="D")
            start += pd.to_timedelta(visit_idx * 3, unit="D")
            stop = start + pd.to_timedelta(int(rng.integers(1, 6)), unit="h")
            rows.append(
                {
                    "Id": f"E{patient.Id[1:]}_{visit_idx + 1:03d}",
                    "START": start.isoformat(),
                    "STOP": stop.isoformat(),
                    "PATIENT": patient.Id,
                    "CODE": "185349003",
                    "DESCRIPTION": "Encounter for check up",
                }
            )
    return pd.DataFrame(rows, columns=["Id", "START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])


def make_conditions(profile: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create condition rows including chronic disease and CVD labels."""
    rows: list[dict[str, object]] = []
    background_conditions = [
        ("195967001", "Asthma"),
        ("13645005", "Hyperlipidemia"),
        ("49727002", "Cough"),
        ("444814009", "Viral sinusitis"),
        ("709044004", "Prediabetes"),
    ]

    for patient in profile.itertuples(index=False):
        assigned: list[tuple[str, str]] = []
        if patient.sim_diabetes:
            assigned.append(("44054006", "Diabetes mellitus"))
        if patient.sim_hypertension:
            assigned.append(("38341003", "Hypertension"))
        if patient.sim_obesity:
            assigned.append(("162864005", "Obesity"))
        if patient.sim_cvd:
            assigned.append(("53741008", "Coronary artery disease"))

        extra_count = int(rng.integers(0, 3))
        for idx in range(extra_count):
            assigned.append(background_conditions[(idx + int(patient.sim_age)) % len(background_conditions)])

        for code, description in assigned:
            start = REFERENCE_DATE - pd.to_timedelta(int(rng.integers(30, 1500)), unit="D")
            stop = pd.NA if rng.random() < 0.88 else (start + pd.to_timedelta(int(rng.integers(20, 365)), unit="D")).date().isoformat()
            rows.append(
                {
                    "START": start.date().isoformat(),
                    "STOP": stop,
                    "PATIENT": patient.Id,
                    "CODE": code,
                    "DESCRIPTION": description,
                }
            )

    return pd.DataFrame(rows, columns=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])


def make_medications(profile: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create medication rows related to the synthetic chronic conditions."""
    rows: list[dict[str, object]] = []
    optional_meds = [
        ("617311", "Atorvastatin 20 MG Oral Tablet"),
        ("197361", "Amlodipine 5 MG Oral Tablet"),
        ("198440", "Acetaminophen 325 MG Oral Tablet"),
    ]

    for patient in profile.itertuples(index=False):
        medications: list[tuple[str, str]] = []
        if patient.sim_diabetes:
            medications.append(("860975", "Metformin 500 MG Oral Tablet"))
        if patient.sim_hypertension:
            medications.append(("617314", "Lisinopril 10 MG Oral Tablet"))
        if patient.sim_cvd:
            medications.extend(
                [
                    ("617311", "Atorvastatin 20 MG Oral Tablet"),
                    ("198211", "Aspirin 81 MG Oral Tablet"),
                ]
            )
        if rng.random() < 0.25:
            medications.append(optional_meds[int(rng.integers(0, len(optional_meds)))])

        for code, description in medications:
            start = REFERENCE_DATE - pd.to_timedelta(int(rng.integers(10, 850)), unit="D")
            stop = start + pd.to_timedelta(int(rng.integers(30, 240)), unit="D")
            rows.append(
                {
                    "START": start.date().isoformat(),
                    "STOP": stop.date().isoformat(),
                    "PATIENT": patient.Id,
                    "CODE": code,
                    "DESCRIPTION": description,
                }
            )

    return pd.DataFrame(rows, columns=["START", "STOP", "PATIENT", "CODE", "DESCRIPTION"])


def make_observations(profile: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create observation rows with BMI values and a few basic vitals."""
    rows: list[dict[str, object]] = []
    for patient in profile.itertuples(index=False):
        measurement_count = int(rng.integers(1, 4))
        for _ in range(measurement_count):
            obs_date = REFERENCE_DATE - pd.to_timedelta(int(rng.integers(5, 720)), unit="D")
            rows.append(
                {
                    "DATE": obs_date.date().isoformat(),
                    "PATIENT": patient.Id,
                    "CODE": "39156-5",
                    "DESCRIPTION": "Body mass index (BMI) [Ratio]",
                    "VALUE": round(float(patient.sim_bmi + rng.normal(0, 0.9)), 1),
                    "UNITS": "kg/m2",
                }
            )
            rows.append(
                {
                    "DATE": obs_date.date().isoformat(),
                    "PATIENT": patient.Id,
                    "CODE": "8480-6",
                    "DESCRIPTION": "Systolic blood pressure",
                    "VALUE": round(float(108 + 0.55 * patient.sim_age + 13 * patient.sim_hypertension + rng.normal(0, 8)), 1),
                    "UNITS": "mmHg",
                }
            )
    return pd.DataFrame(rows, columns=["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE", "UNITS"])


def inject_quality_issues(
    patients: pd.DataFrame,
    conditions: pd.DataFrame,
    medications: pd.DataFrame,
    encounters: pd.DataFrame,
    observations: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Add intentional data quality issues for MediCheck quality-check demos."""
    if not observations.empty:
        bmi_mask = observations["CODE"].eq("39156-5")
        bmi_indices = observations.index[bmi_mask].to_numpy()
        missing_count = max(1, int(len(bmi_indices) * 0.025))
        outlier_count = max(1, int(len(bmi_indices) * 0.006))

        missing_indices = rng.choice(bmi_indices, size=min(missing_count, len(bmi_indices)), replace=False)
        observations.loc[missing_indices, "VALUE"] = np.nan

        available_for_outliers = np.setdiff1d(bmi_indices, missing_indices)
        outlier_indices = rng.choice(
            available_for_outliers,
            size=min(outlier_count, len(available_for_outliers)),
            replace=False,
        )
        observations.loc[outlier_indices, "VALUE"] = 95.0

    if not conditions.empty:
        birthdate_lookup = patients.set_index("Id")["BIRTHDATE"].to_dict()
        error_count = max(1, int(len(conditions) * 0.005))
        error_indices = rng.choice(conditions.index.to_numpy(), size=min(error_count, len(conditions)), replace=False)
        for row_idx in error_indices:
            patient_id = conditions.at[row_idx, "PATIENT"]
            birthdate = pd.Timestamp(birthdate_lookup[patient_id])
            conditions.at[row_idx, "START"] = (birthdate - pd.to_timedelta(int(rng.integers(30, 365)), unit="D")).date().isoformat()

        missing_code_count = max(1, int(len(conditions) * 0.01))
        missing_code_indices = rng.choice(
            conditions.index.to_numpy(),
            size=min(missing_code_count, len(conditions)),
            replace=False,
        )
        conditions.loc[missing_code_indices, "CODE"] = pd.NA

        duplicate_count = max(1, int(len(conditions) * 0.003))
        duplicate_indices = rng.choice(conditions.index.to_numpy(), size=min(duplicate_count, len(conditions)), replace=False)
        conditions = pd.concat([conditions, conditions.loc[duplicate_indices]], ignore_index=True)

    if not medications.empty:
        missing_code_count = max(1, int(len(medications) * 0.006))
        missing_code_indices = rng.choice(
            medications.index.to_numpy(),
            size=min(missing_code_count, len(medications)),
            replace=False,
        )
        medications.loc[missing_code_indices, "CODE"] = pd.NA

    if not encounters.empty:
        duplicate_count = max(1, int(len(encounters) * 0.002))
        duplicate_indices = rng.choice(encounters.index.to_numpy(), size=min(duplicate_count, len(encounters)), replace=False)
        encounters = pd.concat([encounters, encounters.loc[duplicate_indices]], ignore_index=True)

    return {
        "patients": patients,
        "conditions": conditions,
        "medications": medications,
        "encounters": encounters,
        "observations": observations,
    }


def generate_sample_data(
    n_patients: int = DEFAULT_N_PATIENTS,
    out_dir: str | Path = "data/sample",
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """Generate all MediCheck synthetic sample tables as pandas DataFrames."""
    rng = np.random.default_rng(seed)
    profile = build_patient_profile(n_patients=n_patients, seed=seed)
    patients = make_patients(profile)
    tables = inject_quality_issues(
        patients=patients,
        conditions=make_conditions(profile, rng),
        medications=make_medications(profile, rng),
        encounters=make_encounters(profile, rng),
        observations=make_observations(profile, rng),
        rng=rng,
    )
    write_tables(tables, Path(out_dir))
    return tables


def write_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> dict[str, Path]:
    """Write generated MediCheck sample tables to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    file_paths = {
        "patients": out_dir / "patients.csv",
        "conditions": out_dir / "conditions.csv",
        "medications": out_dir / "medications.csv",
        "encounters": out_dir / "encounters.csv",
        "observations": out_dir / "observations.csv",
    }
    for table_name, path in file_paths.items():
        tables[table_name].to_csv(path, index=False)
    return file_paths


def print_generation_summary(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Print generated file paths and row counts."""
    out_path = Path(out_dir)
    for table_name in ["patients", "conditions", "medications", "encounters", "observations"]:
        path = out_path / f"{table_name}.csv"
        print(f"{path} rows={len(tables[table_name])}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sample data generator."""
    parser = argparse.ArgumentParser(description="Generate MediCheck synthetic sample CSV data.")
    parser.add_argument("--n-patients", type=int, default=DEFAULT_N_PATIENTS, help="Number of synthetic patients to create.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"), help="Directory where CSV files will be written.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible data generation.")
    return parser.parse_args()


def main() -> None:
    """Run the MediCheck sample data generator from the command line."""
    args = parse_args()
    tables = generate_sample_data(n_patients=args.n_patients, out_dir=args.out_dir, seed=args.seed)
    print_generation_summary(tables, args.out_dir)


if __name__ == "__main__":
    main()
