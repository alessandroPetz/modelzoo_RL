from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, model_validator

THIS_FILE = Path(__file__).resolve()
RL_DIR = THIS_FILE.parent
try:
    REPO_ROOT = THIS_FILE.parents[5]
except IndexError:
    REPO_ROOT = RL_DIR

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:
    gym = None
    spaces = None

try:
    import torch as th
except Exception as exc:
    raise RuntimeError("PyTorch is required by this RL model.") from exc

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception:
    PPO = None
    DummyVecEnv = None
    VecNormalize = None

    class BaseCallback:
        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose
            self.locals: Dict[str, Any] = {}

from seanergys_modelzoo.dataloader.seanergys_dataloader import SeanergysDataloader
from seanergys_modelzoo.datasets.pm100 import PM100Dataset

try:
    from seanergys_modelzoo.models.common.seanergys_model import (
        SeanergysModel,
        SeanergysModelTask,
    )
    from seanergys_modelzoo.models.common.seanergys_model_metadata import (
        SeanergysModelMetadata,
    )
except Exception:
    import logging
    from abc import ABC, abstractmethod
    from datetime import datetime
    from enum import Enum
    from pydantic import BaseModel, ConfigDict

    class SeanergysModelTask(Enum):
        CLASSIFICATION = 1
        REGRESSION = 2

    class SeanergysModelMetadata(BaseModel):
        name: str
        version: str = "1.0"
        description: Optional[str] = None
        author: Optional[str] = None
        created_at: datetime = Field(default_factory=datetime.utcnow)
        updated_at: Optional[datetime] = None
        framework: Optional[str] = None
        task: Optional[str] = None
        tags: List[str] = Field(default_factory=list)
        metrics: Dict[str, float] = Field(default_factory=dict)
        hyperparameters: Dict[str, object] = Field(default_factory=dict)
        input_schema: Optional[dict] = None
        output_schema: Optional[dict] = None
        is_active: bool = True

        def _touch(self) -> None:
            self.updated_at = datetime.utcnow()

        def set_field(self, field: str, value: Any) -> None:
            if hasattr(self, field):
                setattr(self, field, value)
            else:
                self.__dict__.update({field: value})
            self._touch()

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "SeanergysModelMetadata":
            known_fields = cls.model_fields.keys()
            return cls(**{k: v for k, v in data.items() if k in known_fields})

    class SeanergysModel(BaseModel, ABC):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

        task_type: SeanergysModelTask = Field(...)
        supported_datasets: Optional[List] = Field(default_factory=list)
        is_trained: Optional[bool] = Field(default=False)
        logger: Optional[Any] = Field(default=None)
        metadata: Optional[SeanergysModelMetadata | dict] = Field(default=None)

        def model_post_init(self, __context: Any) -> None:
            if self.metadata is None:
                self.metadata = SeanergysModelMetadata(name=self.__class__.__name__)
            elif isinstance(self.metadata, dict):
                self.metadata = SeanergysModelMetadata.from_dict(self.metadata)
            if self.logger is None:
                self.logger = logging.getLogger(self.__class__.__name__)
                if not self.logger.handlers:
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                    self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        @abstractmethod
        def build_model(self) -> Any:
            raise NotImplementedError

        @abstractmethod
        def train(self, train_data_loader: SeanergysDataloader, val_data_loader: Optional[SeanergysDataloader] = None, training_loss: Optional[Callable] = None, validation_metrics: Optional[List[Callable]] = None, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError

        @abstractmethod
        def predict(self, data_loader: SeanergysDataloader, **kwargs: Any) -> Iterable:
            raise NotImplementedError

        @abstractmethod
        def evaluate(self, data_loader: SeanergysDataloader, metrics: Optional[List[Callable]] = None, **kwargs: Any) -> Any:
            raise NotImplementedError

        @abstractmethod
        def save(self, path: Union[str, Path], **kwargs: Any) -> bool:
            raise NotImplementedError

        @classmethod
        @abstractmethod
        def load(cls, path: Union[str, Path], **kwargs: Any) -> "SeanergysModel":
            raise NotImplementedError

        @classmethod
        def get_train_config(cls, dataset_name: str, get_dummy_data: bool = False, **kwargs: Any):
            raise NotImplementedError

        @classmethod
        def get_test_config(cls, model: "SeanergysModel", dataset_name: str, get_dummy_data: bool = False, **kwargs: Any):
            raise NotImplementedError

        def _save_metadata(self, path: Union[str, Path]) -> None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path = path.parent / f"{path.stem}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as file_handle:
                json.dump(self.metadata.model_dump(mode="json"), file_handle, indent=2)

try:
    SeanergysModelMetadata.model_rebuild(force=True)
except Exception:
    pass
try:
    SeanergysModel.model_rebuild(force=True)
except Exception:
    pass

try:
    from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.utils.battery_model import Battery
    from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.utils.co2 import CarbonIntensityModels
    from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.utils.generator_model import Generator
    from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.utils.price_model import PriceModel
    from seanergys_modelzoo.models.tasks.power_consumption_manage.RL.utils.renewable_real import RenewableModels
except Exception:
    from utils.battery_model import Battery
    from utils.co2 import CarbonIntensityModels
    from utils.generator_model import Generator
    from utils.price_model import PriceModel
    from utils.renewable_real import RenewableModels

DEFAULT_PM100_PATH = RL_DIR / "csvs" / "job_table.parquet"
DEFAULT_CARBON_CSV_PATH = RL_DIR / "csvs" / "carbon_intensity_IT-NORTH-2020.csv"
DEFAULT_OPENMETEO_CACHE_PATH = RL_DIR / "csvs" / "openmeteo_cache_bologna.csv"
DEFAULT_OUTPUT_DIR = RL_DIR / "saved_models" / "rl_agent_sim3_hard_constraint"

RESULTS_FILENAME = "results_sim3_hardconstraint.csv"
MODEL_BASENAME = "ppo_powercap"
VECNORMALIZE_FILENAME = "vecnormalize.pkl"
STATE_FILENAME = "model_state.json"

SOLAR_COST_EUR_PER_KWH = 0.05
WIND_COST_EUR_PER_KWH = 0.04
SOLAR_CO2_G_PER_KWH = 50.0
WIND_CO2_G_PER_KWH = 34.0

ALPHA_COST = 1.0
ALPHA_CO2 = 0.001
ALPHA_VIOLATION = 50.0
ALPHA_SOC_FINAL = 0.0


def require_rl_dependencies() -> None:
    missing: List[str] = []
    if gym is None or spaces is None:
        missing.append("gymnasium")
    if PPO is None or DummyVecEnv is None or VecNormalize is None:
        missing.append("stable-baselines3")
    if missing:
        raise ImportError("This RL agent requires: " + ", ".join(missing))


def parse_utc_timestamp(value: Union[str, pd.Timestamp]) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def resample_list(values: Iterable[Any], n_points: int) -> np.ndarray:
    values_list = list(values) if values is not None else []

    if n_points <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(values_list) == 0:
        return np.zeros(n_points, dtype=np.float64)
    if len(values_list) == 1:
        return np.full(n_points, float(values_list[0]), dtype=np.float64)

    values_np = np.asarray(values_list, dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, len(values_np))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, values_np).astype(np.float64)


def build_episode_windows(
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    episode_hours: Optional[int],
    stride_hours: Optional[int],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if window_end <= window_start:
        raise ValueError("window_end must be greater than window_start")

    total_hours = max(int(np.ceil((window_end - window_start).total_seconds() / 3600.0)), 1)
    resolved_episode_hours = total_hours if episode_hours is None else int(episode_hours)
    resolved_stride_hours = resolved_episode_hours if stride_hours is None else int(stride_hours)

    if resolved_episode_hours <= 0 or resolved_stride_hours <= 0:
        raise ValueError("episode_hours and stride_hours must be positive")

    episode_delta = pd.Timedelta(hours=resolved_episode_hours)
    stride_delta = pd.Timedelta(hours=resolved_stride_hours)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current_start = window_start

    while current_start < window_end:
        current_end = current_start + episode_delta
        if current_end > window_end:
            break
        windows.append((current_start, current_end))
        current_start = current_start + stride_delta

    if not windows:
        windows = [(window_start, window_end)]

    return windows


def ensure_time_power_dataframe(
    df_raw: pd.DataFrame,
    time_column: str = "time",
    power_column: str = "power",
) -> pd.DataFrame:
    df = df_raw.copy()
    if time_column != "time":
        df = df.rename(columns={time_column: "time"})
    if power_column != "power":
        df = df.rename(columns={power_column: "power"})

    required = {"time", "power"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["time", "power"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["power"] = pd.to_numeric(df["power"], errors="coerce").fillna(0.0).astype(np.float64)
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return df


def build_zero_power_episode(
    start: pd.Timestamp,
    end: pd.Timestamp,
    step_seconds: int,
) -> pd.DataFrame:
    step = pd.Timedelta(seconds=max(int(step_seconds), 1))
    time_index = pd.date_range(
        start=start,
        end=end,
        freq=step,
        inclusive="left",
        tz="UTC",
    )
    if len(time_index) == 0:
        time_index = pd.DatetimeIndex([start])
    return pd.DataFrame({"time": time_index, "power": np.zeros(len(time_index), dtype=np.float64)})


def build_prices_safely(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    try:
        price_model = PriceModel()
        price_base, price_high = price_model.prices_from_df(df)
        base = pd.Series(price_base, index=df.index)
        high = pd.Series(price_high, index=df.index)
        base = pd.to_numeric(base, errors="raise").astype(np.float64)
        high = pd.to_numeric(high, errors="raise").astype(np.float64)
        return base, high
    except Exception:
        ts = pd.to_datetime(df["time"], utc=True)
        hour = ts.dt.hour
        is_weekend = ts.dt.dayofweek >= 5

        base = np.where(
            is_weekend,
            0.00009,
            np.where(((hour >= 8) & (hour < 19)), 0.00012, 0.00008),
        )
        high = base * 1.8
        return (
            pd.Series(base, index=df.index, dtype=np.float64),
            pd.Series(high, index=df.index, dtype=np.float64),
        )


def aggregate_pm100_jobs_to_cluster(
    jobs_df: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    step_seconds: int = 20,
) -> pd.DataFrame:
    step = pd.Timedelta(seconds=max(int(step_seconds), 1))
    time_index = pd.date_range(
        start=window_start,
        end=window_end,
        freq=step,
        inclusive="left",
        tz="UTC",
    )

    if len(time_index) == 0:
        return build_zero_power_episode(window_start, window_end, step_seconds)

    if jobs_df.empty:
        return pd.DataFrame({"time": time_index, "power": np.zeros(len(time_index), dtype=np.float64)})

    mask = (jobs_df["start_time"] < window_end) & (jobs_df["end_time"] >= window_start)
    overlapping_jobs = jobs_df.loc[mask]

    if overlapping_jobs.empty:
        return pd.DataFrame({"time": time_index, "power": np.zeros(len(time_index), dtype=np.float64)})

    per_job_frames: List[pd.DataFrame] = []

    for _, row in overlapping_jobs.iterrows():
        start = parse_utc_timestamp(row["start_time"])
        end = parse_utc_timestamp(row["end_time"])
        values = row["node_power_consumption"]

        raw_len = len(values) if values is not None else 0
        n_points = max(raw_len, 1)
        node_power = resample_list(values, n_points)
        timeline = pd.date_range(start=start, end=end, periods=n_points, tz="UTC")

        job_df = pd.DataFrame({"time": timeline, "power": node_power})
        job_df = job_df[(job_df["time"] >= window_start) & (job_df["time"] < window_end)]
        if not job_df.empty:
            per_job_frames.append(job_df)

    if not per_job_frames:
        return pd.DataFrame({"time": time_index, "power": np.zeros(len(time_index), dtype=np.float64)})

    df_all = pd.concat(per_job_frames, ignore_index=True)
    df_cluster = (
        df_all
        .set_index("time")
        .resample(step)
        .sum()
        .reindex(time_index, fill_value=0.0)
        .reset_index()
        .rename(columns={"index": "time"})
    )
    return ensure_time_power_dataframe(df_cluster)


def build_env_dataframe(
    df_raw: pd.DataFrame,
    p_max_rens: Tuple[float, float],
    carbon_csv_path: Union[str, Path],
    weather_cache_path: Union[str, Path],
    default_step_seconds: int = 20,
) -> pd.DataFrame:
    required_columns = {"time", "power"}
    missing = required_columns - set(df_raw.columns)
    if missing:
        raise ValueError(f"Raw episode dataframe is missing columns: {sorted(missing)}")

    df = ensure_time_power_dataframe(df_raw)

    dt_hours = df["time"].diff().dt.total_seconds() / 3600.0
    valid_dt = dt_hours[dt_hours > 0]
    dt_nominal = float(valid_dt.iloc[0]) if len(valid_dt) > 0 else (default_step_seconds / 3600.0)
    df["dt_hours"] = dt_hours.fillna(dt_nominal).astype(np.float64)

    df["price_base"], df["price_high"] = build_prices_safely(df)

    weather_cache_path = Path(weather_cache_path)
    weather_cache_path.parent.mkdir(parents=True, exist_ok=True)

    renewable_model = RenewableModels(seed=42, cache_file=str(weather_cache_path))
    df["P_wind"] = renewable_model.wind_from_openmeteo(df, p_max_rens[0])
    df["P_solar"] = renewable_model.solar_from_openmeteo(df, p_max_rens[1])
    df["P_ren"] = df["P_solar"] + df["P_wind"]

    co2_model = CarbonIntensityModels(csv_file=str(carbon_csv_path))
    df["co2_intensity"] = co2_model.co2_from_csv(df)

    steps_per_hour = max(int(round(1.0 / dt_nominal)), 1)
    df["E_ren_step"] = df["P_ren"] * df["dt_hours"]

    df["forecast_E_ren_1h"] = (
        df["E_ren_step"].rolling(window=steps_per_hour, min_periods=1).sum().shift(-steps_per_hour).fillna(0.0)
    )
    df["forecast_E_ren_6h"] = (
        df["E_ren_step"].rolling(window=steps_per_hour * 6, min_periods=1).sum().shift(-(steps_per_hour * 6)).fillna(0.0)
    )

    co2_mean = float(df["co2_intensity"].mean()) if len(df) > 0 else 0.0
    df["forecast_co2_intensity_1h"] = (
        df["co2_intensity"].rolling(window=steps_per_hour, min_periods=1).mean().shift(-steps_per_hour).fillna(co2_mean)
    )
    df["forecast_co2_intensity_6h"] = (
        df["co2_intensity"].rolling(window=steps_per_hour * 6, min_periods=1).mean().shift(-(steps_per_hour * 6)).fillna(co2_mean)
    )

    numeric_cols = [
        "power",
        "dt_hours",
        "price_base",
        "price_high",
        "P_wind",
        "P_solar",
        "P_ren",
        "co2_intensity",
        "E_ren_step",
        "forecast_E_ren_1h",
        "forecast_E_ren_6h",
        "forecast_co2_intensity_1h",
        "forecast_co2_intensity_6h",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    fallback_dt = float(default_step_seconds) / 3600.0

    df["power"] = df["power"].fillna(0.0).clip(lower=0.0)
    df["dt_hours"] = df["dt_hours"].fillna(fallback_dt).clip(lower=1e-9)

    df["price_base"] = df["price_base"].fillna(0.0001).clip(lower=0.0)
    df["price_high"] = df["price_high"].fillna(df["price_base"] * 1.5).clip(lower=0.0)

    df["P_wind"] = df["P_wind"].fillna(0.0)
    df["P_solar"] = df["P_solar"].fillna(0.0)
    df["P_ren"] = df["P_ren"].fillna(df["P_wind"] + df["P_solar"]).clip(lower=0.0)

    co2_fallback = df["co2_intensity"].median() if df["co2_intensity"].notna().any() else 300.0
    df["co2_intensity"] = df["co2_intensity"].fillna(co2_fallback).clip(lower=0.0)

    df["E_ren_step"] = df["E_ren_step"].fillna(0.0)
    df["forecast_E_ren_1h"] = df["forecast_E_ren_1h"].fillna(0.0)
    df["forecast_E_ren_6h"] = df["forecast_E_ren_6h"].fillna(0.0)
    df["forecast_co2_intensity_1h"] = df["forecast_co2_intensity_1h"].fillna(df["co2_intensity"])
    df["forecast_co2_intensity_6h"] = df["forecast_co2_intensity_6h"].fillna(df["co2_intensity"])

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def prepare_output_dir(output_dir: Union[str, Path]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / RESULTS_FILENAME
    with open(results_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(
            "episodio;total_cost;total_co2_kg;SOC_finale;capacity_ratio;"
            "battery_use_Wh;gen_use_Wh;violation_Wh\n"
        )


def identity_collate(batch: List[Any]) -> Any:
    return batch[0]


def _unwrap_singleton(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        if flat.size == 1:
            return flat[0]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def extract_episode_df(sample: Any) -> pd.DataFrame:
    if isinstance(sample, pd.DataFrame):
        return sample.reset_index(drop=True)

    if isinstance(sample, tuple) and len(sample) >= 1:
        first = _unwrap_singleton(sample[0])
        if isinstance(first, pd.DataFrame):
            return first.reset_index(drop=True)

    raise TypeError(
        "Expected a pandas DataFrame or a tuple whose first element contains a DataFrame. "
        f"Found {type(sample)} instead."
    )


GymBaseEnv = gym.Env if gym is not None else object


class HPCBatteryEnv(GymBaseEnv):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        episode_loader: Optional[SeanergysDataloader] = None,
        threshold: float = 400000,
        battery_capacity: float = 3200000,
        max_charge_rate: float = 3200000,
        max_discharge_rate: float = 3200000,
        generator_max_power_w: float = 500000,
        generator_min_power_w: float = 50000,
        generator_efficiency: float = 0.4,
        generator_fuel_cost_per_wh: float = 0.00025,
        generator_co2_g_per_kwh: float = 450.0,
        p_max_rens: Tuple[float, float] = (400000, 450000),
        carbon_csv_path: Union[str, Path] = DEFAULT_CARBON_CSV_PATH,
        weather_cache_path: Union[str, Path] = DEFAULT_OPENMETEO_CACHE_PATH,
        step_seconds: int = 20,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        save_episode_artifacts: bool = True,
    ):
        require_rl_dependencies()
        super().__init__()

        if (df is None) == (episode_loader is None):
            raise ValueError("Pass exactly one between `df` and `episode_loader`.")

        self.base_raw_df = None if df is None else df.reset_index(drop=True).copy()
        self.episode_loader = episode_loader
        self.loader_iter = iter(episode_loader) if episode_loader is not None else None

        self.threshold = float(threshold)
        self.capacity = float(battery_capacity)
        self.max_charge_rate = float(max_charge_rate)
        self.max_discharge_rate = float(max_discharge_rate)

        self.generator_max_power_w = float(generator_max_power_w)
        self.generator_min_power_w = float(generator_min_power_w)
        self.generator_efficiency = float(generator_efficiency)
        self.generator_fuel_cost_per_wh = float(generator_fuel_cost_per_wh)
        self.generator_co2_g_per_kwh = float(generator_co2_g_per_kwh)

        self.p_max_rens = tuple(float(x) for x in p_max_rens)
        self.carbon_csv_path = str(carbon_csv_path)
        self.weather_cache_path = str(weather_cache_path)
        self.step_seconds = int(step_seconds)
        self.output_dir = Path(output_dir)
        self.save_episode_artifacts = bool(save_episode_artifacts)

        self.action_levels = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([len(self.action_levels), len(self.action_levels)])
        self.observation_space = spaces.Box(
            low=np.array([0., 0., 0., 0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0.], dtype=np.float32),
            high=np.array([3., 3., 1., 1., 1., 1., 1., 1., 1., 3., 3., 3., 1., 1.], dtype=np.float32),
            dtype=np.float32,
        )

        self.episode_idx = -1
        self.df = pd.DataFrame()
        self.N = 0
        self.price_base_max = 1.0
        self.price_high_max = 1.0
        self.co2_min = 0.0
        self.co2_max = 1.0
        self._current_episode_saved = False
        self.reset_logs()

    def reset_logs(self) -> None:
        self.battery_history: List[float] = []
        self.battery_use_history: List[float] = []
        self.gen_use_history: List[float] = []
        self.time_history: List[pd.Timestamp] = []
        self.cost_history: List[float] = []
        self.co2_history: List[float] = []
        self.curtailment_history: List[float] = []
        self.violation_history: List[float] = []

        self.grid_cost_history: List[float] = []
        self.grid_co2_history: List[float] = []
        self.gen_cost_history: List[float] = []
        self.gen_co2_history: List[float] = []
        self.green_cost_history: List[float] = []
        self.green_co2_history: List[float] = []
        self.batt_cost_history: List[float] = []
        self.batt_co2_history: List[float] = []

    def _next_raw_episode_df(self) -> pd.DataFrame:
        if self.episode_loader is None:
            return self.base_raw_df.copy() if self.base_raw_df is not None else pd.DataFrame()

        assert self.loader_iter is not None
        try:
            sample = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.episode_loader)
            sample = next(self.loader_iter)
        return extract_episode_df(sample)

    def _prepare_episode_df(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        processed_cols = {
            "price_base",
            "price_high",
            "P_wind",
            "P_solar",
            "P_ren",
            "co2_intensity",
            "forecast_E_ren_1h",
            "forecast_E_ren_6h",
            "forecast_co2_intensity_1h",
            "forecast_co2_intensity_6h",
            "dt_hours",
        }
        if processed_cols.issubset(df_raw.columns):
            df = df_raw.copy()
            df["time"] = pd.to_datetime(df["time"], utc=True)
            return df.sort_values("time").reset_index(drop=True)

        return build_env_dataframe(
            df_raw=df_raw,
            p_max_rens=self.p_max_rens,
            carbon_csv_path=self.carbon_csv_path,
            weather_cache_path=self.weather_cache_path,
            default_step_seconds=self.step_seconds,
        )

    def _load_episode(self) -> None:
        raw_df = self._next_raw_episode_df()
        if raw_df.empty:
            now = pd.Timestamp.now(tz="UTC")
            raw_df = build_zero_power_episode(now, now + pd.Timedelta(seconds=self.step_seconds), self.step_seconds)

        self.df = self._prepare_episode_df(raw_df)
        self.N = len(self.df)
        self.price_base_max = max(float(self.df["price_base"].max()), 1e-9)
        self.price_high_max = max(float(self.df["price_high"].max()), 1e-9)
        self.co2_min = float(self.df["co2_intensity"].min())
        self.co2_max = float(self.df["co2_intensity"].max())
        self._current_episode_saved = False

    def _get_obs(self) -> np.ndarray:
        t = min(self.t, self.N - 1)

        p_load = float(self.df.loc[t, "power"])
        p_ren = float(self.df.loc[t, "P_ren"])
        p_deficit_after_ren = max(p_load - p_ren, 0.0)

        p_ratio = np.clip(p_load / (self.threshold + 1e-9), 0.0, 3.0)
        deficit_after_ren_ratio = np.clip(p_deficit_after_ren / (self.threshold + 1e-9), 0.0, 3.0)
        battery_norm = float(self.battery.energy / max(self.battery.capacity, 1e-9))

        ts = self.df.loc[t, "time"]
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        price_base = float(self.df.loc[t, "price_base"])
        price_high = float(self.df.loc[t, "price_high"])
        price_base_norm = price_base / self.price_base_max
        price_high_norm = price_high / self.price_high_max

        co2_int = float(self.df.loc[t, "co2_intensity"])
        co2_norm = (co2_int - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)

        time_left = 1.0 - (t / max(self.N - 1, 1))
        p_ren_norm = np.clip(p_ren / (self.threshold + 1e-9), 0.0, 3.0)

        e_forecast_1h = float(self.df.loc[t, "forecast_E_ren_1h"])
        e_forecast_6h = float(self.df.loc[t, "forecast_E_ren_6h"])
        e_forecast_1h_norm = np.clip(e_forecast_1h / (self.capacity + 1e-9), 0.0, 3.0)
        e_forecast_6h_norm = np.clip(e_forecast_6h / (6.0 * self.capacity + 1e-9), 0.0, 3.0)

        co2_forecast_1h = float(self.df.loc[t, "forecast_co2_intensity_1h"])
        co2_forecast_6h = float(self.df.loc[t, "forecast_co2_intensity_6h"])
        co2_forecast_1h_norm = (co2_forecast_1h - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)
        co2_forecast_6h_norm = (co2_forecast_6h - self.co2_min) / (self.co2_max - self.co2_min + 1e-9)

        obs = np.array(
            [
                p_ratio,
                deficit_after_ren_ratio,
                battery_norm,
                time_left,
                price_base_norm,
                price_high_norm,
                co2_norm,
                hour_sin,
                hour_cos,
                p_ren_norm,
                e_forecast_1h_norm,
                e_forecast_6h_norm,
                co2_forecast_1h_norm,
                co2_forecast_6h_norm,
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _save_previous_episode(self) -> None:
        if not self.save_episode_artifacts:
            return
        if self.episode_idx < 0 or self._current_episode_saved:
            return
        if len(self.time_history) == 0:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.output_dir / RESULTS_FILENAME
        end_soc = self.battery.info()["SOC"]
        end_capacity_ratio = self.battery.info()["capacity_Wh"] / self.capacity

        with open(csv_path, "a", encoding="utf-8") as file_handle:
            file_handle.write(
                f"{self.episode_idx};"
                f"{sum(self.cost_history):.4f};"
                f"{sum(self.co2_history) / 1000.0:.4f};"
                f"{end_soc:.4f};"
                f"{end_capacity_ratio:.4f};"
                f"{sum(self.battery_use_history):.3f};"
                f"{sum(self.gen_use_history):.3f};"
                f"{sum(self.violation_history):.3f}\n"
            )
        self._current_episode_saved = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._save_previous_episode()
        self._load_episode()

        self.episode_idx += 1
        self.reset_logs()
        self.t = 0

        self.battery = Battery(
            capacity_wh=self.capacity,
            initial_charge_wh=self.capacity / 2.0,
            max_charge_rate_w=self.max_charge_rate,
            max_discharge_rate_w=self.max_discharge_rate,
        )
        self.generator = Generator(
            max_power_w=self.generator_max_power_w,
            min_power_w=self.generator_min_power_w,
            efficiency=self.generator_efficiency,
            fuel_cost_per_wh=self.generator_fuel_cost_per_wh,
            co2_g_per_kwh=self.generator_co2_g_per_kwh,
        )

        self.battery_history = [self.battery.energy]
        self.time_history = [self.df.loc[0, "time"]]
        self.cost_history = [0.0]
        self.co2_history = [0.0]
        self.gen_use_history = [0.0]
        self.battery_use_history = [0.0]
        self.curtailment_history = [0.0]
        self.violation_history = [0.0]

        self.grid_cost_history = [0.0]
        self.grid_co2_history = [0.0]
        self.gen_cost_history = [0.0]
        self.gen_co2_history = [0.0]
        self.green_cost_history = [0.0]
        self.green_co2_history = [0.0]
        self.batt_cost_history = [0.0]
        self.batt_co2_history = [0.0]

        obs = self._get_obs()
        return obs, {}

    def step(self, action: Union[np.ndarray, List[int], Tuple[int, int]]):
        idx_discharge, idx_gen_peak = [int(x) for x in np.asarray(action).reshape(-1)[:2]]
        a_discharge = float(self.action_levels[idx_discharge])
        a_gen_peak = float(self.action_levels[idx_gen_peak])

        t = self.t
        dt = float(self.df.loc[t, "dt_hours"])

        if dt <= 0:
            self.t += 1
            terminated = self.t >= self.N
            obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated and self.save_episode_artifacts:
                self._save_previous_episode()
            return obs, 0.0, terminated, False, {}

        p_load = float(self.df.loc[t, "power"])
        p_wind = float(self.df.loc[t, "P_wind"])
        p_solar = float(self.df.loc[t, "P_solar"])
        p_ren = float(self.df.loc[t, "P_ren"])

        price_base = float(self.df.loc[t, "price_base"])
        co2_intensity = float(self.df.loc[t, "co2_intensity"])

        e_wind_total = p_wind * dt
        e_solar_total = p_solar * dt

        green_cost = (
            (e_wind_total / 1000.0) * WIND_COST_EUR_PER_KWH +
            (e_solar_total / 1000.0) * SOLAR_COST_EUR_PER_KWH
        )
        green_co2_g = (
            (e_wind_total / 1000.0) * WIND_CO2_G_PER_KWH +
            (e_solar_total / 1000.0) * SOLAR_CO2_G_PER_KWH
        )

        p_from_ren = min(p_load, p_ren)
        p_surplus = max(p_ren - p_load, 0.0)
        p_deficit = max(p_load - p_ren, 0.0)

        e_surplus = p_surplus * dt
        e_deficit = p_deficit * dt

        e_charged = 0.0
        e_curtail = e_surplus
        if e_surplus > 0:
            e_charged = self.battery.charge(p_surplus, dt)
            e_curtail = max(e_surplus - e_charged, 0.0)

        e_from_batt = 0.0
        if e_deficit > 0 and a_discharge > 0:
            requested_batt = a_discharge * e_deficit
            e_from_batt = self.battery.discharge(requested_batt / dt, dt)
            e_from_batt = min(e_from_batt, e_deficit)

        e_after_batt = max(e_deficit - e_from_batt, 0.0)

        e_from_gen = 0.0
        if e_after_batt > 0 and a_gen_peak > 0:
            requested_gen = a_gen_peak * e_after_batt
            p_requested_gen = requested_gen / dt
            e_from_gen = self.generator.dispatch(p_requested_gen, dt)
            e_from_gen = min(e_from_gen, e_after_batt)

        e_after_gen = max(e_after_batt - e_from_gen, 0.0)

        p_grid = e_after_gen / dt if dt > 0 else 0.0
        p_grid_allowed = min(p_grid, self.threshold)
        p_grid_violation = max(p_grid - self.threshold, 0.0)

        e_from_grid = p_grid_allowed * dt
        e_violation = p_grid_violation * dt

        cost_grid = e_from_grid * price_base
        cost_gen = self.generator.get_cost(e_from_gen)

        e_batt_throughput = e_charged + e_from_batt
        batt_cost = self.battery.step_cost(dt, e_batt_throughput)

        step_cost = green_cost + batt_cost + cost_grid + cost_gen

        co2_grid = (e_from_grid / 1000.0) * co2_intensity
        co2_gen = self.generator.get_co2(e_from_gen / 1000.0)
        batt_co2_g = self.battery.step_co2_g(dt, e_batt_throughput)
        step_co2_g = green_co2_g + batt_co2_g + co2_grid + co2_gen

        violation_kwh = e_violation / 1000.0
        reward = (
            - ALPHA_COST * step_cost
            - ALPHA_CO2 * step_co2_g
            - ALPHA_VIOLATION * violation_kwh
        )

        last_step = self.t == self.N - 1
        if last_step and ALPHA_SOC_FINAL > 0:
            reward = (
                - sum(self.cost_history)
                - ALPHA_VIOLATION * sum(self.violation_history)
            )

        self.cost_history.append(step_cost)
        self.co2_history.append(step_co2_g)
        self.gen_use_history.append(e_from_gen)
        self.battery_use_history.append(e_from_batt)
        self.curtailment_history.append(e_curtail)
        self.violation_history.append(e_violation)

        self.grid_cost_history.append(cost_grid)
        self.grid_co2_history.append(co2_grid)
        self.gen_cost_history.append(cost_gen)
        self.gen_co2_history.append(co2_gen)
        self.green_cost_history.append(green_cost)
        self.green_co2_history.append(green_co2_g)
        self.batt_cost_history.append(batt_cost)
        self.batt_co2_history.append(batt_co2_g)

        self.battery.step(dt)
        self.battery_history.append(self.battery.energy)
        self.time_history.append(self.df.loc[t, "time"])

        self.t += 1
        terminated = self.t >= self.N
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs()

        info = {
            "P_from_ren": p_from_ren,
            "step_cost": step_cost,
            "step_co2_g": step_co2_g,
            "E_from_grid": e_from_grid,
            "E_from_gen": e_from_gen,
            "E_from_batt": e_from_batt,
            "E_to_batt": e_charged,
            "E_violation": e_violation,
            "E_curtail": e_curtail,
        }

        if terminated and self.save_episode_artifacts:
            self._save_previous_episode()

        return obs, float(reward), terminated, False, info

    def close(self) -> None:
        if self.save_episode_artifacts:
            self._save_previous_episode()
        return None


class StopAfterNEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_episodes = int(max_episodes)
        self.episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.episode_count += int(np.sum(dones))
            if self.episode_count >= self.max_episodes:
                if self.verbose:
                    print(f"Stopping training after {self.episode_count} episodes")
                return False
        return True


class RLAgentSim3HardConstraint(SeanergysModel):
    pm100_path: Optional[Union[str, Path]] = Field(default=(str(DEFAULT_PM100_PATH) if DEFAULT_PM100_PATH.exists() else None))
    use_zenodo_url: bool = Field(default=(not DEFAULT_PM100_PATH.exists()))
    is_dummy: bool = Field(default=False)

    window_start: Union[str, pd.Timestamp] = Field(default="2020-05-25T00:00:00Z")
    window_end: Union[str, pd.Timestamp] = Field(default="2020-06-25T00:00:00Z")
    episode_hours: Optional[int] = Field(default=24)
    stride_hours: Optional[int] = Field(default=24)
    step_seconds: int = Field(default=20)

    threshold: float = Field(default=400000)
    battery_capacity: float = Field(default=3_200_000)
    p_max_wind: float = Field(default=400000)
    p_max_solar: float = Field(default=450000)
    generator_max_power_w: float = Field(default=500000)
    generator_min_power_w: float = Field(default=50000)
    generator_efficiency: float = Field(default=0.4)
    generator_fuel_cost_per_wh: float = Field(default=0.00025)
    generator_co2_g_per_kwh: float = Field(default=450.0)

    carbon_csv_path: Union[str, Path] = Field(default=str(DEFAULT_CARBON_CSV_PATH))
    weather_cache_path: Union[str, Path] = Field(default=str(DEFAULT_OPENMETEO_CACHE_PATH))
    output_dir: Union[str, Path] = Field(default=str(DEFAULT_OUTPUT_DIR))

    learning_rate: float = Field(default=1e-4)
    n_steps: int = Field(default=4096)
    batch_size: int = Field(default=512)
    gae_lambda: float = Field(default=0.92)
    ent_coef: float = Field(default=0.01)
    clip_range: float = Field(default=0.15)
    max_grad_norm: float = Field(default=0.5)
    total_timesteps: int = Field(default=10_000_000)
    max_episodes: int = Field(default=2)
    device: str = Field(default="cpu")
    seed: Optional[int] = Field(default=42)

    policy_net_arch: List[int] = Field(default_factory=lambda: [256, 256, 128])
    normalize_observations: bool = Field(default=True)
    normalize_rewards: bool = Field(default=True)
    clip_reward: float = Field(default=10.0)

    model_basename: str = Field(default=MODEL_BASENAME)
    vecnormalize_filename: str = Field(default=VECNORMALIZE_FILENAME)
    training_history: Dict[str, Any] = Field(default_factory=dict)

    rl_model: Optional[Any] = Field(default=None, exclude=True)
    vec_normalize: Optional[Any] = Field(default=None, exclude=True)
    _last_train_loader: Optional[SeanergysDataloader] = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _inject_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values.setdefault("task_type", SeanergysModelTask.REGRESSION)
        values.setdefault("supported_datasets", [PM100Dataset])
        values.setdefault(
            "metadata",
            SeanergysModelMetadata(
                name="RLAgentSim3HardConstraint",
                created_at=pd.Timestamp.utcnow().to_pydatetime(),
                framework="stable-baselines3",
                task="reinforcement_learning",
                description="PPO controller using PM100Dataset preprocessing to build RL episodes.",
                tags=["rl", "ppo", "power-cap", "battery", "generator", "pm100"],
            ),
        )
        return values

    @model_validator(mode="after")
    def validate_model(self) -> "RLAgentSim3HardConstraint":
        if self.pm100_path is None and not self.use_zenodo_url and DEFAULT_PM100_PATH.exists():
            self.pm100_path = str(DEFAULT_PM100_PATH)
        if self.pm100_path is None and not self.use_zenodo_url:
            self.use_zenodo_url = True
        return self

    def build_model(self) -> "RLAgentSim3HardConstraint":
        return self

    def build_pm100_episode_rows(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        start = parse_utc_timestamp(self.window_start)
        end = parse_utc_timestamp(self.window_end)

        jobs = jobs_df.copy()
        if not jobs.empty:
            jobs["start_time"] = pd.to_datetime(jobs["start_time"], utc=True)
            jobs["end_time"] = pd.to_datetime(jobs["end_time"], utc=True)

        cluster_df = aggregate_pm100_jobs_to_cluster(
            jobs_df=jobs,
            window_start=start,
            window_end=end,
            step_seconds=self.step_seconds,
        )

        episodes = build_episode_windows(start, end, self.episode_hours, self.stride_hours)
        rows: List[Dict[str, Any]] = []

        for episode_index, (ep_start, ep_end) in enumerate(episodes):
            episode_df = cluster_df[(cluster_df["time"] >= ep_start) & (cluster_df["time"] < ep_end)].copy()
            if episode_df.empty:
                episode_df = build_zero_power_episode(ep_start, ep_end, self.step_seconds)

            rows.append(
                {
                    "episode_df": episode_df.reset_index(drop=True),
                    "episode_meta": {
                        "episode_index": episode_index,
                        "window_start": ep_start.isoformat(),
                        "window_end": ep_end.isoformat(),
                    },
                }
            )

        if not rows:
            rows.append(
                {
                    "episode_df": build_zero_power_episode(start, end, self.step_seconds),
                    "episode_meta": {
                        "episode_index": 0,
                        "window_start": start.isoformat(),
                        "window_end": end.isoformat(),
                    },
                }
            )

        return pd.DataFrame(rows)

    def build_episode_dataset(self, **overrides: Any) -> PM100Dataset:
        params = {
            "pm100_path": self.pm100_path,
            "use_zenodo_url": self.use_zenodo_url,
            "is_dummy": self.is_dummy,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }
        params.update(overrides)

        start = parse_utc_timestamp(params["window_start"])
        end = parse_utc_timestamp(params["window_end"])

        return PM100Dataset(
            data_path=params["pm100_path"],
            use_zenodo_url=bool(params["use_zenodo_url"]),
            is_dummy=bool(params["is_dummy"]),
            transform=None,
            target_transform=None,
            preprocessing_functions=[self.build_pm100_episode_rows],
            filters=[
                ("start_time", "<", end),
                ("end_time", ">=", start),
            ],
            columns=["start_time", "end_time", "node_power_consumption"],
            input_features=["episode_df"],
            output_features=["episode_meta"],
        )

    def build_dataloader(
        self,
        dataset: Optional[PM100Dataset] = None,
        shuffle: bool = True,
        **dataset_overrides: Any,
    ) -> SeanergysDataloader:
        resolved_dataset = dataset if dataset is not None else self.build_episode_dataset(**dataset_overrides)
        return SeanergysDataloader(
            dataset=resolved_dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=identity_collate,
            num_workers=0,
        )

    def _build_env(
        self,
        episode_loader: Optional[SeanergysDataloader] = None,
        df: Optional[pd.DataFrame] = None,
        save_episode_artifacts: bool = True,
    ) -> HPCBatteryEnv:
        return HPCBatteryEnv(
            df=df,
            episode_loader=episode_loader,
            threshold=self.threshold,
            battery_capacity=self.battery_capacity,
            max_charge_rate=self.battery_capacity,
            max_discharge_rate=self.battery_capacity,
            generator_max_power_w=self.generator_max_power_w,
            generator_min_power_w=self.generator_min_power_w,
            generator_efficiency=self.generator_efficiency,
            generator_fuel_cost_per_wh=self.generator_fuel_cost_per_wh,
            generator_co2_g_per_kwh=self.generator_co2_g_per_kwh,
            p_max_rens=(self.p_max_wind, self.p_max_solar),
            carbon_csv_path=self.carbon_csv_path,
            weather_cache_path=self.weather_cache_path,
            step_seconds=self.step_seconds,
            output_dir=self.output_dir,
            save_episode_artifacts=save_episode_artifacts,
        )

    def _build_policy_kwargs(self) -> Dict[str, Any]:
        return {"net_arch": list(self.policy_net_arch), "activation_fn": th.nn.ReLU}

    def _attach_or_create_vecnormalize(self, vec_env: Any, training: bool) -> Any:
        if self.vec_normalize is not None:
            self.vec_normalize.set_venv(vec_env)
            self.vec_normalize.training = training
            self.vec_normalize.norm_reward = self.normalize_rewards if training else False
            return self.vec_normalize

        return VecNormalize(
            vec_env,
            norm_obs=self.normalize_observations,
            norm_reward=self.normalize_rewards if training else False,
            clip_reward=self.clip_reward,
            training=training,
        )

    def _normalize_obs_for_policy(self, obs: np.ndarray) -> np.ndarray:
        if self.vec_normalize is None:
            return obs
        if hasattr(self.vec_normalize, "normalize_obs"):
            normalized = self.vec_normalize.normalize_obs(np.array(obs, copy=True))
            return np.asarray(normalized)
        return obs

    def train(
        self,
        train_data_loader: SeanergysDataloader,
        val_data_loader: Optional[SeanergysDataloader] = None,
        training_loss: Optional[Callable] = None,
        validation_metrics: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del training_loss, validation_metrics
        require_rl_dependencies()

        total_timesteps = int(kwargs.get("total_timesteps", self.total_timesteps))
        max_episodes = int(kwargs.get("max_episodes", self.max_episodes))
        learning_rate = float(kwargs.get("learning_rate", self.learning_rate))
        n_steps = int(kwargs.get("n_steps", self.n_steps))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        gae_lambda = float(kwargs.get("gae_lambda", self.gae_lambda))
        ent_coef = float(kwargs.get("ent_coef", self.ent_coef))
        clip_range = float(kwargs.get("clip_range", self.clip_range))
        max_grad_norm = float(kwargs.get("max_grad_norm", self.max_grad_norm))
        verbose = int(kwargs.get("verbose", 0))
        reset_model = bool(kwargs.get("reset_model", False))
        reset_output_dir = bool(kwargs.get("reset_output_dir", True))
        reset_num_timesteps = bool(kwargs.get("reset_num_timesteps", True))
        save_episode_artifacts = bool(kwargs.get("save_episode_artifacts", True))

        if reset_output_dir:
            prepare_output_dir(self.output_dir)

        base_env = self._build_env(
            episode_loader=train_data_loader,
            save_episode_artifacts=save_episode_artifacts,
        )
        vec_env = DummyVecEnv([lambda: base_env])
        vec_env = self._attach_or_create_vecnormalize(vec_env, training=True)

        training_start = time.time()
        if self.rl_model is None or reset_model:
            self.rl_model = PPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs=self._build_policy_kwargs(),
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                clip_range=clip_range,
                max_grad_norm=max_grad_norm,
                verbose=verbose,
                device=self.device,
                seed=self.seed,
            )
        else:
            self.rl_model.set_env(vec_env)

        stop_callback = StopAfterNEpisodes(max_episodes=max_episodes, verbose=verbose)
        self.rl_model.learn(
            total_timesteps=total_timesteps,
            callback=stop_callback,
            reset_num_timesteps=reset_num_timesteps,
        )

        if hasattr(base_env, "close"):
            base_env.close()
        if hasattr(vec_env, "close"):
            vec_env.close()

        training_time = time.time() - training_start
        self.vec_normalize = vec_env if isinstance(vec_env, VecNormalize) else None
        self._last_train_loader = train_data_loader
        self.is_trained = True

        train_summary = {
            "total_timesteps": total_timesteps,
            "max_episodes": max_episodes,
            "training_time_seconds": training_time,
            "output_dir": str(self.output_dir),
        }

        validation_summary = self.evaluate(val_data_loader) if val_data_loader is not None else None
        self.training_history = {"training": train_summary, "validation": validation_summary}

        self.metadata.set_field("last_trained", pd.Timestamp.utcnow().isoformat())
        self.metadata.set_field("training_time_seconds", training_time)
        self.metadata.set_field(
            "hyperparameters",
            {
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gae_lambda": gae_lambda,
                "ent_coef": ent_coef,
                "clip_range": clip_range,
                "max_grad_norm": max_grad_norm,
                "threshold": self.threshold,
                "battery_capacity": self.battery_capacity,
            },
        )
        return self.training_history

    def predict(
        self,
        data_loader: Union[SeanergysDataloader, pd.DataFrame, Iterable[Any]],
        deterministic: bool = True,
        return_trajectory: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        del kwargs
        require_rl_dependencies()
        if self.rl_model is None:
            raise RuntimeError("The PPO model is not available. Train or load the model before calling predict().")

        results: List[Dict[str, Any]] = []
        iterator: Iterable[Any] = [data_loader] if isinstance(data_loader, pd.DataFrame) else data_loader

        for episode_idx, sample in enumerate(iterator):
            episode_df = extract_episode_df(sample)
            env = self._build_env(df=episode_df, save_episode_artifacts=False)
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            trajectory: List[Dict[str, Any]] = []

            while not done:
                model_obs = self._normalize_obs_for_policy(obs)
                action, _ = self.rl_model.predict(model_obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)

                if return_trajectory:
                    trajectory.append(
                        {
                            "action": np.asarray(action).reshape(-1).tolist(),
                            **{
                                k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                for k, v in info.items()
                            },
                        }
                    )

            summary = self._summarize_episode(env, episode_idx, episode_reward)
            if return_trajectory:
                summary["trajectory"] = trajectory
            results.append(summary)
            env.close()

        return results

    def evaluate(
        self,
        data_loader: SeanergysDataloader,
        metrics: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        episode_results = self.predict(data_loader, deterministic=True, return_trajectory=False)
        if len(episode_results) == 0:
            return {"n_episodes": 0}

        summary = {
            "n_episodes": len(episode_results),
            "mean_reward": float(np.mean([item["episode_reward"] for item in episode_results])),
            "mean_total_cost": float(np.mean([item["total_cost"] for item in episode_results])),
            "mean_total_co2_kg": float(np.mean([item["total_co2_kg"] for item in episode_results])),
            "mean_violation_wh": float(np.mean([item["violation_wh"] for item in episode_results])),
            "mean_end_soc": float(np.mean([item["end_soc"] for item in episode_results])),
            "mean_battery_use_wh": float(np.mean([item["battery_use_wh"] for item in episode_results])),
            "mean_generator_use_wh": float(np.mean([item["generator_use_wh"] for item in episode_results])),
            "episodes": episode_results,
        }

        custom_metrics: Dict[str, Any] = {}
        if metrics:
            for metric in metrics:
                metric_name = getattr(metric, "__name__", metric.__class__.__name__)
                try:
                    custom_metrics[metric_name] = metric(episode_results)
                except Exception as exc:
                    custom_metrics[metric_name] = f"Metric execution failed: {exc}"
        if custom_metrics:
            summary["custom_metrics"] = custom_metrics

        self.metadata.set_field(
            "metrics",
            {
                "mean_reward": summary["mean_reward"],
                "mean_total_cost": summary["mean_total_cost"],
                "mean_total_co2_kg": summary["mean_total_co2_kg"],
                "mean_violation_wh": summary["mean_violation_wh"],
            },
        )
        return summary

    def save(self, path: Union[str, Path] = DEFAULT_OUTPUT_DIR, **kwargs: Any) -> bool:
        del kwargs
        require_rl_dependencies()
        if self.rl_model is None:
            raise RuntimeError("There is no trained PPO model to save.")

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / self.model_basename
        vec_path = save_dir / self.vecnormalize_filename
        state_path = save_dir / STATE_FILENAME

        self.rl_model.save(str(model_path))
        if self.vec_normalize is not None:
            self.vec_normalize.save(str(vec_path))

        config = self.model_dump(
            mode="json",
            exclude={
                "logger",
                "supported_datasets",
                "rl_model",
                "vec_normalize",
            },
        )
        with open(state_path, "w", encoding="utf-8") as file_handle:
            json.dump(config, file_handle, indent=2)

        self._save_metadata(save_dir / self.model_basename)
        return True

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_OUTPUT_DIR, **kwargs: Any) -> "RLAgentSim3HardConstraint":
        require_rl_dependencies()
        save_dir = Path(path)
        state_path = save_dir / STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(f"Model state file not found: {state_path}")

        with open(state_path, "r", encoding="utf-8") as file_handle:
            config = json.load(file_handle)
        config.update(kwargs)

        obj = cls(**config)
        model_path = save_dir / obj.model_basename
        vec_path = save_dir / obj.vecnormalize_filename

        obj.rl_model = PPO.load(str(model_path), device=obj.device)
        if vec_path.exists():
            with open(vec_path, "rb") as file_handle:
                obj.vec_normalize = pickle.load(file_handle)
                if hasattr(obj.vec_normalize, "training"):
                    obj.vec_normalize.training = False
        obj.is_trained = True
        return obj

    def _summarize_episode(self, env: HPCBatteryEnv, episode_idx: int, episode_reward: float) -> Dict[str, Any]:
        battery_info = env.battery.info()
        return {
            "episode_index": int(episode_idx),
            "episode_reward": float(episode_reward),
            "n_steps": int(env.t),
            "total_cost": float(np.sum(env.cost_history)),
            "total_co2_kg": float(np.sum(env.co2_history) / 1000.0),
            "violation_wh": float(np.sum(env.violation_history)),
            "battery_use_wh": float(np.sum(env.battery_use_history)),
            "generator_use_wh": float(np.sum(env.gen_use_history)),
            "curtailment_wh": float(np.sum(env.curtailment_history)),
            "end_soc": float(battery_info["SOC"]),
            "end_capacity_ratio": float(battery_info["capacity_Wh"] / env.capacity),
        }

    @classmethod
    def get_train_config(
        cls,
        dataset_name: str = "pm100",
        get_dummy_data: bool = False,
        **kwargs: Any,
    ) -> Tuple["RLAgentSim3HardConstraint", SeanergysDataloader]:
        if dataset_name.lower().strip() != "pm100":
            raise ValueError("This version supports only dataset_name='pm100'.")

        params = dict(kwargs)
        params.setdefault("is_dummy", get_dummy_data)
        if "pm100_path" not in params:
            params["pm100_path"] = str(DEFAULT_PM100_PATH) if DEFAULT_PM100_PATH.exists() else None
        if "use_zenodo_url" not in params:
            params["use_zenodo_url"] = params.get("pm100_path") is None

        params.setdefault("window_start", "2020-05-25T00:00:00Z")
        params.setdefault("window_end", "2020-06-25T00:00:00Z")
        params.setdefault("episode_hours", 24)
        params.setdefault("stride_hours", 24)

        model = cls(**params)
        dataloader = model.build_dataloader(shuffle=True)
        return model, dataloader

    @classmethod
    def get_test_config(
        cls,
        model: "RLAgentSim3HardConstraint",
        dataset_name: str = "pm100",
        get_dummy_data: bool = False,
        **kwargs: Any,
    ) -> SeanergysDataloader:
        if dataset_name.lower().strip() != "pm100":
            raise ValueError("This version supports only dataset_name='pm100'.")

        params = {
            "pm100_path": model.pm100_path,
            "use_zenodo_url": model.use_zenodo_url,
            "is_dummy": get_dummy_data,
            "window_start": model.window_start,
            "window_end": model.window_end,
            "episode_hours": model.episode_hours,
            "stride_hours": model.stride_hours,
            "step_seconds": model.step_seconds,
        }
        params.update(kwargs)

        return model.build_dataloader(shuffle=False, **params)


RLAgentSim3HardConstraint.model_rebuild(force=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seanergys-integrated RL agent for power consumption management using PM100Dataset"
    )

    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--pm100-path", default=str(DEFAULT_PM100_PATH))
    parser.add_argument("--use-zenodo-url", action="store_true")
    parser.add_argument("--is-dummy", action="store_true")

    parser.add_argument("--window-start", default="2020-05-25T00:00:00Z")
    parser.add_argument("--window-end", default="2020-06-25T00:00:00Z")
    parser.add_argument("--episode-hours", type=int, default=24)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--step-seconds", type=int, default=20)

    parser.add_argument("--threshold", type=float, default=400000)
    parser.add_argument("--battery-capacity", type=float, default=3_200_000)
    parser.add_argument("--p-max-wind", type=float, default=400000)
    parser.add_argument("--p-max-solar", type=float, default=450000)
    parser.add_argument("--generator-max-power-w", type=float, default=500000)
    parser.add_argument("--generator-min-power-w", type=float, default=50000)
    parser.add_argument("--generator-efficiency", type=float, default=0.4)
    parser.add_argument("--generator-fuel-cost-per-wh", type=float, default=0.00025)
    parser.add_argument("--generator-co2-g-per-kwh", type=float, default=450.0)

    parser.add_argument("--carbon-csv-path", default=str(DEFAULT_CARBON_CSV_PATH))
    parser.add_argument("--weather-cache-path", default=str(DEFAULT_OPENMETEO_CACHE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--save-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--load-dir", default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gae-lambda", type=float, default=0.92)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-episode-artifacts", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pm100_path = None if args.use_zenodo_url else args.pm100_path

    common_kwargs = {
        "pm100_path": pm100_path,
        "use_zenodo_url": args.use_zenodo_url,
        "is_dummy": args.is_dummy,
        "window_start": args.window_start,
        "window_end": args.window_end,
        "episode_hours": args.episode_hours,
        "stride_hours": args.stride_hours,
        "step_seconds": args.step_seconds,
        "threshold": args.threshold,
        "battery_capacity": args.battery_capacity,
        "p_max_wind": args.p_max_wind,
        "p_max_solar": args.p_max_solar,
        "generator_max_power_w": args.generator_max_power_w,
        "generator_min_power_w": args.generator_min_power_w,
        "generator_efficiency": args.generator_efficiency,
        "generator_fuel_cost_per_wh": args.generator_fuel_cost_per_wh,
        "generator_co2_g_per_kwh": args.generator_co2_g_per_kwh,
        "carbon_csv_path": args.carbon_csv_path,
        "weather_cache_path": args.weather_cache_path,
        "output_dir": args.output_dir,
        "total_timesteps": args.total_timesteps,
        "max_episodes": args.max_episodes,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "clip_range": args.clip_range,
        "max_grad_norm": args.max_grad_norm,
        "device": args.device,
    }

    if args.mode == "train":
        model, train_loader = RLAgentSim3HardConstraint.get_train_config(
            dataset_name="pm100",
            get_dummy_data=args.is_dummy,
            **common_kwargs,
        )
        history = model.train(train_loader, save_episode_artifacts=args.save_episode_artifacts)
        model.save(args.save_dir)
        print(json.dumps(history, indent=2, default=str))
        print(f"Model saved to: {args.save_dir}")
        return

    model = RLAgentSim3HardConstraint.load(args.load_dir, **common_kwargs)
    test_loader = RLAgentSim3HardConstraint.get_test_config(
        model=model,
        dataset_name="pm100",
        get_dummy_data=args.is_dummy,
        **common_kwargs,
    )

    if args.mode == "evaluate":
        results = model.evaluate(test_loader)
    else:
        results = model.predict(test_loader, deterministic=True, return_trajectory=False)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
