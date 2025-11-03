"""Streamlit dashboard for TÜİK Address-Based Population Registration data."""
from __future__ import annotations

import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# -----------------------------------------------------------------------------
# Streamlit & Altair configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="TÜİK Nüfus ve Demografi Paneli",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("default")
alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BULLETIN_URL = (
    "https://data.tuik.gov.tr/Bulten/Index?"
    "p=Address-Based-Population-Registration-System-Results-2023-49685"
)

TABLE_IDS = {
    "population_timeseries": 1590,
    "district_population": 2305,
    "median_age": 2306,
    "age_dependency": 2307,
    "foreign_population": 2881,
    "household_size": 2308,
    "growth_density": 1591,
    "population_age_group": 945,
    "marital_status": 2741,
    "single_age_population": 2820,
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
}

SEX_MAP = {
    "toplam": "Toplam",
    "toplamtotal": "Toplam",
    "total": "Toplam",
    "turkiye": "Toplam",
    "erkek": "Erkek",
    "erkekmale": "Erkek",
    "male": "Erkek",
    "kadin": "Kadin",
    "kadn": "Kadin",
    "kadun": "Kadin",
    "kadinfemale": "Kadin",
    "female": "Kadin",
}


_TURKISH_TRANSLATION = str.maketrans(
    {
        "İ": "I",
        "ı": "i",
        "Ş": "S",
        "ş": "s",
        "Ü": "U",
        "ü": "u",
        "Ö": "O",
        "ö": "o",
        "Ç": "C",
        "ç": "c",
        "Ğ": "G",
        "ğ": "g",
    }
)


SEX_ORDER = ["Kadin", "Erkek", "Toplam"]
SEX_COLOR_SCALE = alt.Scale(domain=SEX_ORDER, range=["#c03a7b", "#1f77b4", "#4d4d4d"])
SEX_COLOR_LEGEND = alt.Legend(title="Cinsiyet", labelExpr="replace(datum.label, 'Kadin', 'Kadın')")

SEX_TWO_SCALE = alt.Scale(domain=["Kadin", "Erkek"], range=["#c03a7b", "#1f77b4"])
SEX_TWO_LEGEND = alt.Legend(title="Cinsiyet", labelExpr="replace(datum.label, 'Kadin', 'Kad\u0131n')")


def clean_text(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\xa0", " ")
        .strip()
    )
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("dipnot") or lowered.startswith("not"):
        return None
    if lowered in {"toplam-total", "toplam", "total"}:
        return "Turkiye"
    text = text.translate(_TURKISH_TRANSLATION)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    return ascii_text or None


def normalize_sex(label: object) -> Optional[str]:
    text = clean_text(label)
    if text is None:
        return None
    key = re.sub(r"[^a-z]", "", text.lower())
    if key in SEX_MAP:
        return SEX_MAP[key]
    return text


YEAR_PATTERN = re.compile(r"(20\d{2})")


def parse_year(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    text = clean_text(value)
    if text is None:
        return None
    match = YEAR_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return None


def simplify_status(label: object) -> Optional[str]:
    text = clean_text(label)
    if text is None:
        return None
    primary = text.split("-")[0].strip()
    return primary or text


def slugify_label(label: object) -> str:
    text = clean_text(label) or str(label)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text.lower())
    return ascii_text.strip()


def age_group_sort_key(label: object) -> Tuple[int, str]:
    if label is None:
        return (999, "")
    text = str(label)
    lowered = text.lower()
    if lowered.startswith("toplam"):
        return (-1, text)
    match = re.match(r"(\d+)", lowered)
    if match:
        return (int(match.group(1)), text)
    match = re.match(r"(\d+)", text)
    if match:
        return (int(match.group(1)), text)
    return (999, text)


def ensure_cache_file(path: Path, fetcher) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fetcher()
    return path


# -----------------------------------------------------------------------------
# TÜİK bulletin download helpers
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner="TÜİK meta verisi indiriliyor...")
def get_bulletin_meta() -> Dict:
    response = requests.get(BULLETIN_URL, headers=DEFAULT_HEADERS, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    script = soup.find("script", id="bultendata")
    if not script or not script.string:
        raise RuntimeError("TÜİK meta verisi bulunamadı.")
    payload = script.string.strip()
    if payload.startswith("window.bultendata="):
        payload = payload[len("window.bultendata="):]
    payload = payload.rstrip(";")
    return json.loads(payload)


def resolve_table_meta(table_id: int) -> Dict:
    meta = get_bulletin_meta()
    for cluster in meta.get("IstatistikselTablolar", []):
        for upper in cluster.get("KatUstler", []):
            for sub in upper.get("Katlar", []):
                for table in sub.get("IstatistikselTablolar", []):
                    if table.get("ItId") == table_id:
                        return table
    raise RuntimeError(f"TÜİK tablosu bulunamadı: {table_id}")


def download_table(table_id: int) -> Path:
    table_meta = resolve_table_meta(table_id)
    target = CACHE_DIR / f"tuik_{table_id}.xls"

    def fetch():
        url = "https://data.tuik.gov.tr/Bulten/DownloadIstatistikselTablo?p=" + table_meta["DosyaAdi"]
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)

    return ensure_cache_file(target, fetch)


# -----------------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------------


def _read_raw_excel(table_id: int) -> pd.DataFrame:
    path = download_table(table_id)
    return pd.read_excel(path, header=None)


@st.cache_data(show_spinner="Yıllara göre nüfus serileri yükleniyor...")
def load_population_timeseries() -> pd.DataFrame:
    path = download_table(TABLE_IDS["population_timeseries"])
    wide = pd.read_excel(path, header=2)
    years = [
        int(float(year))
        for year in wide.iloc[0, 1:].tolist()
        if not pd.isna(year)
    ]
    data = wide.iloc[1:].copy()
    data.columns = ["province"] + years
    data["province"] = data["province"].apply(clean_text)
    data = data.dropna(subset=["province"])
    numeric_cols = data.columns[1:]
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    long_df = data.melt(id_vars="province", var_name="year", value_name="population")
    long_df = long_df.dropna(subset=["population"])
    long_df["year"] = long_df["year"].astype(int)
    long_df["population"] = long_df["population"].astype(float)
    return long_df


@st.cache_data(show_spinner="İl/ilçe nüfusu yükleniyor...")
def load_population_settlement() -> pd.DataFrame:
    population_provinces = set(load_population_timeseries()["province"].unique())
    raw = pd.read_excel(download_table(TABLE_IDS["district_population"]), header=5)
    raw = raw.rename(
        columns={
            raw.columns[0]: "area_name",
            raw.columns[1]: "population_total",
            raw.columns[2]: "population_urban",
            raw.columns[3]: "population_rural",
            raw.columns[4]: "unused",
            raw.columns[5]: "annual_growth_rate",
        }
    )
    raw = raw.drop(columns=["unused"])
    raw["population_urban"] = raw["population_urban"].replace("-", pd.NA)
    raw["population_rural"] = raw["population_rural"].replace("-", pd.NA)
    for col in ["population_total", "population_urban", "population_rural", "annual_growth_rate"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    rows: List[Dict[str, object]] = []
    current_province: Optional[str] = None
    for record in raw.to_dict("records"):
        area = clean_text(record["area_name"])
        if area is None:
            continue
        if area == "Türkiye":
            area_type = "country"
            current_province = "Türkiye"
        elif area in population_provinces:
            area_type = "province"
            current_province = area
        else:
            area_type = "district"
        province = current_province if current_province else area
        rows.append(
            {
                "province": province,
                "area_name": area,
                "area_type": area_type,
                "population_total": record["population_total"],
                "population_urban": record["population_urban"],
                "population_rural": record["population_rural"],
                "annual_growth_rate": record["annual_growth_rate"],
            }
        )
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["population_total"])
    return df


def _extract_year_sex_matrix(
    table_id: int,
    data_start_row: int = 3,
    year_row_index: int = 1,
    sex_row_index: int = 2,
) -> Tuple[pd.Series, pd.DataFrame, List[int], List[str]]:
    raw = _read_raw_excel(table_id)
    province_series = raw.iloc[data_start_row:, 0].apply(clean_text)
    year_row = raw.iloc[year_row_index, 1:]
    sex_row = raw.iloc[sex_row_index, 1:]
    values = raw.iloc[data_start_row:, 1:]

    years: List[int] = []
    sexes: List[str] = []
    last_year: Optional[int] = None
    for y_raw, s_raw in zip(year_row, sex_row):
        year = None
        if not pd.isna(y_raw):
            try:
                year = int(float(str(y_raw).strip().split()[0]))
            except (ValueError, TypeError):
                year = None
        if year is None:
            year = last_year
        else:
            last_year = year
        sex = normalize_sex(s_raw)
        years.append(year if year is not None else last_year)
        sexes.append(sex if sex is not None else "Toplam")
    return province_series, values, years, sexes


@st.cache_data(show_spinner="Ortanca yaş serileri yükleniyor...")
def load_median_age() -> pd.DataFrame:
    provinces, matrix, years, sexes = _extract_year_sex_matrix(
        TABLE_IDS["median_age"],
        data_start_row=4,
        year_row_index=2,
        sex_row_index=3,
    )
    records: List[Dict[str, object]] = []
    for col_idx, (year, sex) in enumerate(zip(years, sexes)):
        if year is None or sex is None:
            continue
        column_values = pd.to_numeric(matrix.iloc[:, col_idx], errors="coerce")
        for province, value in zip(provinces, column_values):
            if province is None or pd.isna(value):
                continue
            records.append(
                {
                    "province": province,
                    "year": int(year),
                    "sex": sex,
                    "median_age": float(value),
                }
            )
    df = pd.DataFrame(records)
    return df


@st.cache_data(show_spinner="Yaş bağımlılık oranları yükleniyor...")
def load_age_dependency() -> pd.DataFrame:
    raw = _read_raw_excel(TABLE_IDS["age_dependency"])
    data = raw.iloc[3:, :10].copy()
    data.columns = [
        "year",
        "province",
        "population_total",
        "age_0_14",
        "age_15_64",
        "age_65_plus",
        "unused",
        "dependency_total",
        "dependency_child",
        "dependency_elderly",
    ]
    data["year"] = pd.to_numeric(data["year"], errors="coerce").ffill()
    data["province"] = data["province"].apply(clean_text)
    data = data.dropna(subset=["province", "year"])
    for col in [
        "population_total",
        "age_0_14",
        "age_15_64",
        "age_65_plus",
        "dependency_total",
        "dependency_child",
        "dependency_elderly",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["year"] = data["year"].astype(int)
    data = data.dropna(subset=["dependency_total"])
    return data.reset_index(drop=True)


@st.cache_data(show_spinner="Yabancı nüfus verisi yükleniyor...")
def load_foreign_population() -> pd.DataFrame:
    provinces, matrix, years, sexes = _extract_year_sex_matrix(
        TABLE_IDS["foreign_population"],
        data_start_row=4,
        year_row_index=2,
        sex_row_index=3,
    )
    records: List[Dict[str, object]] = []
    for col_idx, (year, sex) in enumerate(zip(years, sexes)):
        if year is None or sex is None:
            continue
        column_values = pd.to_numeric(matrix.iloc[:, col_idx], errors="coerce")
        for province, value in zip(provinces, column_values):
            if province is None or pd.isna(value):
                continue
            records.append(
                {
                    "province": province,
                    "year": int(year),
                    "sex": sex,
                    "foreign_population": float(value),
                }
            )
    df = pd.DataFrame(records)
    return df


@st.cache_data(show_spinner="Hanehalkı büyüklüğü yükleniyor...")
def load_household_size() -> pd.DataFrame:
    raw = _read_raw_excel(TABLE_IDS["household_size"])
    provinces = raw.iloc[3:, 0].apply(clean_text)
    year_row = raw.iloc[2, 1:]
    values = raw.iloc[3:, 1:]
    years: List[int] = []
    last_year: Optional[int] = None
    for value in year_row:
        year = None
        if not pd.isna(value):
            try:
                year = int(float(value))
            except (ValueError, TypeError):
                year = None
        if year is None:
            year = last_year
        else:
            last_year = year
        years.append(year)
    records: List[Dict[str, object]] = []
    for col_idx, year in enumerate(years):
        if year is None:
            continue
        column_values = pd.to_numeric(values.iloc[:, col_idx], errors="coerce")
        for province, value in zip(provinces, column_values):
            if province is None or pd.isna(value):
                continue
            records.append(
                {
                    "province": province,
                    "year": int(year),
                    "average_household_size": float(value),
                }
            )
    return pd.DataFrame(records)


@st.cache_data(show_spinner="Nufus artis hizi ve yogunlugu yukleniyor...")
def load_growth_density() -> pd.DataFrame:
    raw = _read_raw_excel(TABLE_IDS["growth_density"])
    metric_labels = raw.iloc[2, 1:]
    period_labels = raw.iloc[3, 1:]
    values = raw.iloc[4:, 1:]
    provinces = raw.iloc[4:, 0].apply(clean_text)

    records: List[Dict[str, object]] = []
    current_metric = "annual_growth_rate"
    current_period: Optional[str] = None

    for col_idx, (metric_cell, period_cell) in enumerate(zip(metric_labels, period_labels)):
        metric_text = clean_text(metric_cell)
        if metric_text:
            metric_lower = metric_text.lower()
            if "nufus yogunlugu" in metric_lower or "population density" in metric_lower:
                current_metric = "population_density"
            elif "yillik nufus artis hizi" in metric_lower or "annual growth rate" in metric_lower:
                current_metric = "annual_growth_rate"

        period_text = clean_text(period_cell)
        if period_text and re.fullmatch(r"\d+(\.0+)?", period_text):
            period_text = str(int(float(period_text)))
        current_period = period_text or current_period
        if current_period is None:
            continue

        column_values = pd.to_numeric(values.iloc[:, col_idx], errors="coerce")
        for province, value in zip(provinces, column_values):
            if province is None or pd.isna(value):
                continue
            records.append(
                {
                    "province": province,
                    "period": current_period,
                    "metric": current_metric,
                    "value": float(value),
                }
            )
    df = pd.DataFrame(records)
    return df


@st.cache_data(show_spinner="Yaş grubu dağılımları yükleniyor...")
def load_age_group_population() -> pd.DataFrame:
    path = download_table(TABLE_IDS["population_age_group"])
    raw = pd.read_excel(path, header=2)
    year_col, age_col, sex_col = raw.columns[:3]
    province_cols = list(raw.columns[3:])

    raw[year_col] = raw[year_col].ffill()
    raw[age_col] = raw[age_col].ffill()
    raw = raw.dropna(subset=[sex_col])
    raw["year"] = raw[year_col].apply(parse_year)
    raw["age_group"] = raw[age_col].apply(clean_text)
    raw["sex"] = raw[sex_col].apply(normalize_sex)
    raw = raw.dropna(subset=["year", "age_group", "sex"])

    province_map = {col: clean_text(col) or str(col) for col in province_cols}

    melted = raw.melt(
        id_vars=["year", "age_group", "sex"],
        value_vars=province_cols,
        var_name="province_raw",
        value_name="population",
    )
    melted["province"] = melted["province_raw"].map(province_map)
    melted = melted.drop(columns=["province_raw"])
    melted = melted.dropna(subset=["province", "population"])
    melted["population"] = pd.to_numeric(melted["population"], errors="coerce")
    melted = melted.dropna(subset=["population"])
    melted["population"] = melted["population"].astype(float)
    melted["year"] = melted["year"].astype(int)
    melted["age_group"] = melted["age_group"].astype(str)
    return melted.reset_index(drop=True)


@st.cache_data(show_spinner="Medeni durum verileri yükleniyor...")
def load_marital_status() -> pd.DataFrame:
    path = download_table(TABLE_IDS["marital_status"])
    raw = pd.read_excel(path, header=[3, 4])

    normalized_cols: List[Tuple[str, str]] = []
    for primary, secondary in raw.columns:
        primary_clean = clean_text(primary) or str(primary).strip()
        secondary_clean = clean_text(secondary) or str(secondary).strip()
        if secondary_clean.startswith("Unnamed"):
            secondary_clean = ""
        if secondary_clean.endswith(".1"):
            secondary_clean = secondary_clean.replace(".1", "")
        normalized_cols.append((primary_clean, secondary_clean))

    raw.columns = pd.MultiIndex.from_tuples(normalized_cols)
    raw = raw.sort_index(axis=1)

    def find_column(prefixes: Tuple[str, ...]) -> Tuple[str, str]:
        for col in raw.columns:
            slug = slugify_label(col[0])
            if any(slug.startswith(prefix) for prefix in prefixes):
                return col
        raise KeyError(prefixes)

    year_col = find_column(("yl year", "yil", "year"))
    province_col = find_column(("il provinces", "il", "province"))

    year_series = raw[year_col].ffill().iloc[:, 0]
    province_series = raw[province_col].iloc[:, 0].apply(clean_text)

    raw = raw.drop(columns=[year_col, province_col])
    value_part = raw.groupby(level=[0, 1], axis=1).sum()
    value_part.columns = pd.MultiIndex.from_tuples(
        [
            (clean_text(col[0]) or str(col[0]), clean_text(col[1]) or str(col[1]))
            for col in value_part.columns
        ]
    )
    value_part["year"] = year_series.values
    value_part["province"] = province_series.values

    melted = (
        value_part.set_index(["year", "province"])
        .stack(level=[0, 1], future_stack=True)
        .reset_index()
        .rename(
            columns={
                "level_2": "marital_status",
                "level_3": "sex_raw",
                0: "population",
            }
        )
    )
    melted["year"] = melted["year"].apply(parse_year)
    melted = melted.dropna(subset=["year", "province"])
    melted["sex"] = melted["sex_raw"].apply(normalize_sex)
    melted = melted.dropna(subset=["sex"])
    melted["marital_status"] = melted["marital_status"].apply(lambda x: clean_text(x) or x)
    melted = melted[melted["marital_status"].notna()]
    melted["population"] = pd.to_numeric(melted["population"], errors="coerce")
    melted = melted.dropna(subset=["population"])
    melted["population"] = melted["population"].astype(float)
    melted["year"] = melted["year"].astype(int)
    melted["province"] = melted["province"].apply(lambda x: x or "Bilinmeyen")
    return melted.reset_index(drop=True)


@st.cache_data(show_spinner="Tek yaş nüfusları yükleniyor...")
def load_single_age_population() -> pd.DataFrame:
    path = download_table(TABLE_IDS["single_age_population"])
    raw = pd.read_excel(path, header=3)
    raw.columns = ["year", "province", "sex", "total"] + raw.columns[4:].tolist()

    raw["year"] = raw["year"].ffill().apply(parse_year)
    raw["province"] = raw["province"].ffill().apply(clean_text)
    raw["sex"] = raw["sex"].ffill().apply(normalize_sex)

    age_cols = [col for col in raw.columns if col not in {"year", "province", "sex", "total"}]
    melted = raw.melt(
        id_vars=["year", "province", "sex", "total"],
        value_vars=age_cols,
        var_name="age_label",
        value_name="population",
    )
    melted = melted.dropna(subset=["population"])

    def parse_age(value: object) -> Optional[int]:
        if isinstance(value, (int, float)) and not pd.isna(value):
            return int(value)
        text = clean_text(value)
        if text is None:
            return None
        if text.endswith("+"):
            return 75
        try:
            return int(float(text))
        except (ValueError, TypeError):
            return None

    melted["age"] = melted["age_label"].apply(parse_age)
    melted = melted.dropna(subset=["year", "province", "sex", "age"])
    melted["population"] = pd.to_numeric(melted["population"], errors="coerce")
    melted = melted.dropna(subset=["population"])
    melted["population"] = melted["population"].astype(float)
    melted["year"] = melted["year"].astype(int)
    melted["age"] = melted["age"].astype(int)
    return melted.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------


def empty_chart(message: str) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"message": [message]}))
        .mark_text(align="center", baseline="middle", color="#6e6e6e", fontSize=14)
        .encode(text="message:N")
        .properties(height=220)
    )


def population_trend_chart(df: pd.DataFrame, provinces: Iterable[str]) -> alt.Chart:
    subset = df[df["province"].isin(provinces)]
    return (
        alt.Chart(subset)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y(
                "population:Q",
                title="Nüfus",
                axis=alt.Axis(format="~,d", tickMinStep=50000),
            ),
            color=alt.Color("province:N", title="İl"),
            tooltip=[
                alt.Tooltip("province:N", title="İl"),
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("population:Q", title="Nüfus", format=",.0f"),
            ],
        )
        .properties(height=320)
    )


def median_age_chart(df: pd.DataFrame, province: str) -> alt.Chart:
    subset = df[df["province"] == province]
    return (
        alt.Chart(subset)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("median_age:Q", title="Ortanca yaş"),
            color=alt.Color(
                "sex:N",
                scale=SEX_COLOR_SCALE,
                legend=SEX_COLOR_LEGEND,
            ),
            tooltip=[
                alt.Tooltip("sex:N", title="Cinsiyet"),
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("median_age:Q", title="Ortanca yaş", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def foreign_population_chart(df: pd.DataFrame, year: int, top_n: int = 15) -> alt.Chart:
    subset = df[df["year"] == year]
    totals = subset[subset["sex"] == "Toplam"][["province", "foreign_population"]]
    top_provinces = (
        totals.sort_values("foreign_population", ascending=False)
        .head(top_n)["province"]
        .tolist()
    )
    filtered = subset[subset["province"].isin(top_provinces)]
    return (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            y=alt.Y(
                "province:N",
                sort=top_provinces[::-1],
                title="İl",
                axis=alt.Axis(labelLimit=200),
            ),
            x=alt.X("foreign_population:Q", title="Yabancı nüfus"),
            color=alt.Color(
                "sex:N",
                scale=SEX_COLOR_SCALE,
                legend=SEX_COLOR_LEGEND,
            ),
            tooltip=[
                alt.Tooltip("province:N", title="İl"),
                alt.Tooltip("sex:N", title="Cinsiyet"),
                alt.Tooltip("foreign_population:Q", title="Nüfus", format=",.0f"),
            ],
        )
        .properties(height=400)
    )


def dependency_chart(df: pd.DataFrame, province: str) -> alt.Chart:
    subset = df[df["province"] == province]
    melted = subset.melt(
        id_vars=["year"],
        value_vars=["dependency_child", "dependency_elderly", "dependency_total"],
        var_name="ratio_type",
        value_name="value",
    )
    ratio_names = {
        "dependency_child": "Çocuk bağımlılık oranı",
        "dependency_elderly": "Yaşlı bağımlılık oranı",
        "dependency_total": "Toplam bağımlılık oranı",
    }
    melted["ratio_type"] = melted["ratio_type"].map(ratio_names)
    return (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Yıl"),
            y=alt.Y("value:Q", title="Bağımlılık oranı (%)"),
            color=alt.Color("ratio_type:N", title="Gösterge"),
            tooltip=[
                alt.Tooltip("ratio_type:N", title="Gösterge"),
                alt.Tooltip("year:O", title="Yıl"),
                alt.Tooltip("value:Q", title="Oran", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def household_size_chart(df: pd.DataFrame, year: int, top_n: int = 10) -> alt.Chart:
    subset = df[df["year"] == year].copy()
    subset = subset.sort_values("average_household_size", ascending=False).head(top_n)
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            y=alt.Y(
                "province:N",
                sort=subset["average_household_size"].tolist()[::-1],
                title="İl",
                axis=alt.Axis(labelLimit=200),
            ),
            x=alt.X("average_household_size:Q", title="Ortalama hanehalkı büyüklüğü"),
            tooltip=[
                alt.Tooltip("province:N", title="İl"),
                alt.Tooltip("average_household_size:Q", title="Büyüklük", format=".2f"),
            ],
        )
        .properties(height=380)
    )


def growth_density_chart(df: pd.DataFrame, metric: str, year_label: str, top_n: int = 12) -> alt.Chart:
    subset = df[(df["metric"] == metric) & (df["period"] == year_label)]
    subset = subset.sort_values("value", ascending=False).head(top_n)
    metric_title = "Nüfus yoğunluğu (kişi/km²)" if metric == "population_density" else "Yıllık nüfus artış hızı (‰)"
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            y=alt.Y(
                "province:N",
                sort=subset["value"].tolist()[::-1],
                axis=alt.Axis(labelLimit=220),
                title="İl",
            ),
            x=alt.X("value:Q", title=metric_title),
            tooltip=[
                alt.Tooltip("province:N", title="İl"),
                alt.Tooltip("value:Q", title="Değer", format=".2f"),
            ],
        )
        .properties(height=380)
    )


def district_population_chart(df: pd.DataFrame, province: str, top_n: int = 12) -> alt.Chart:
    subset = df[(df["province"] == province) & (df["area_type"] == "district")].copy()
    if subset.empty:
        return empty_chart("İlçe bazında veri bulunamadı.")

    subset["population_urban"] = pd.to_numeric(subset["population_urban"], errors="coerce").fillna(0)
    subset["population_rural"] = pd.to_numeric(subset["population_rural"], errors="coerce").fillna(0)
    subset["total"] = subset["population_urban"] + subset["population_rural"]
    subset = subset.sort_values("total", ascending=False).head(top_n)

    melted = subset.melt(
        id_vars=["area_name", "total"],
        value_vars=["population_urban", "population_rural"],
        var_name="settlement_type",
        value_name="population",
    )
    label_map = {
        "population_urban": "İl/ilçe merkezi",
        "population_rural": "Belde/köy",
    }
    melted["settlement_type"] = melted["settlement_type"].map(label_map)
    order = subset["area_name"].tolist()
    chart_height = max(240, 28 * len(order))

    return (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            y=alt.Y(
                "area_name:N",
                sort=order[::-1],
                title="İlçe",
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X("population:Q", title="Nüfus", axis=alt.Axis(format=",.0f")),
            color=alt.Color("settlement_type:N", title="Yerleşim türü"),
            tooltip=[
                alt.Tooltip("area_name:N", title="İlçe"),
                alt.Tooltip("settlement_type:N", title="Yerleşim"),
                alt.Tooltip("population:Q", title="Nüfus", format=",.0f"),
            ],
        )
        .properties(height=chart_height)
    )


def newborn_chart(df: pd.DataFrame, year: int, top_n: int = 15) -> alt.Chart:
    subset = df[(df["year"] == year) & (df["age"] == 0) & (df["sex"] == "Toplam")].copy()
    if subset.empty:
        return empty_chart("Secilen yil icin dogum verisi bulunamadi.")
    subset = subset.sort_values("population", ascending=False).head(top_n)
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            y=alt.Y(
                "province:N",
                sort=subset["population"].tolist()[::-1],
                axis=alt.Axis(labelLimit=220),
                title="Il",
            ),
            x=alt.X(
                "population:Q",
                title="0 yas nufusu (yaklasik dogum sayisi)",
                axis=alt.Axis(format=",.0f"),
            ),
            tooltip=[
                alt.Tooltip("province:N", title="Il"),
                alt.Tooltip("population:Q", title="Nufus", format=",.0f"),
            ],
        )
        .properties(height=max(280, 26 * len(subset)))
    )


def district_growth_leader_chart(df: pd.DataFrame, top_n: int = 10, *, descending: bool = True) -> alt.Chart:
    subset = df[(df["area_type"] == "district") & df["annual_growth_rate"].notna()].copy()
    if subset.empty:
        return empty_chart("Ilce bazinda nufus artis verisi bulunamadi.")
    subset = subset.sort_values("annual_growth_rate", ascending=not descending).head(top_n)
    order = subset["annual_growth_rate"].tolist()[::-1] if descending else subset["annual_growth_rate"].tolist()
    title = "En hizli buyuyen ilceler" if descending else "En hizli daralan ilceler"
    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            y=alt.Y("area_name:N", sort=order, axis=alt.Axis(labelLimit=220), title="Ilce"),
            x=alt.X("annual_growth_rate:Q", title="Yillik artis hizi (promil)", axis=alt.Axis(format=".2f")),
            color=alt.Color("province:N", title="Il"),
            tooltip=[
                alt.Tooltip("area_name:N", title="Ilce"),
                alt.Tooltip("province:N", title="Il"),
                alt.Tooltip("annual_growth_rate:Q", title="Artis hizi", format=".2f"),
            ],
        )
        .properties(title=title, height=max(260, 24 * len(subset)))
    )


def age_pyramid_chart(df: pd.DataFrame, province: str, year: int) -> alt.Chart:
    subset = df[
        (df["province"] == province)
        & (df["year"] == year)
        & (df["sex"].isin(["Kadin", "Erkek"]))
    ].copy()
    if subset.empty:
        return empty_chart("Seçilen filtreler için yaş grubu verisi bulunamadı.")

    subset["age_group"] = subset["age_group"].astype(str)
    age_order = sorted(subset["age_group"].unique(), key=age_group_sort_key)
    subset["plot_population"] = subset.apply(
        lambda row: -row["population"] if row["sex"] == "Erkek" else row["population"],
        axis=1,
    )
    max_abs = subset["population"].max()
    chart_height = max(260, 26 * len(age_order))

    return (
        alt.Chart(subset)
        .mark_bar()
        .encode(
            y=alt.Y("age_group:N", sort=age_order, title="Yaş grubu"),
            x=alt.X(
                "plot_population:Q",
                title="Nüfus",
                axis=alt.Axis(format=",.0f", labelExpr="abs(datum.value)"),
                scale=alt.Scale(domain=[-max_abs * 1.05, max_abs * 1.05]),
            ),
            color=alt.Color(
                "sex:N",
                scale=SEX_TWO_SCALE,
                legend=SEX_TWO_LEGEND,
            ),
            tooltip=[
                alt.Tooltip("age_group:N", title="Yaş grubu"),
                alt.Tooltip("sex:N", title="Cinsiyet"),
                alt.Tooltip("population:Q", title="Nüfus", format=",.0f"),
            ],
        )
        .properties(height=chart_height)
    )


def marital_status_chart(df: pd.DataFrame, province: str, year: int) -> alt.Chart:
    subset = df[(df["province"] == province) & (df["year"] == year)].copy()
    if subset.empty:
        return empty_chart("Seçilen filtreler için medeni durum verisi bulunamadı.")

    filtered = subset[subset["sex"].isin(["Kadin", "Erkek"])].copy()
    if filtered.empty:
        filtered = subset[subset["sex"] == "Toplam"].copy()
        if filtered.empty:
            return empty_chart("Seçilen filtreler için medeni durum verisi bulunamadı.")

    filtered["status_display"] = filtered["marital_status"].apply(simplify_status)
    filtered = filtered[filtered["status_display"].notna()]
    if filtered.empty:
        return empty_chart("Medeni durum verisi gösterilemiyor.")

    totals = filtered.groupby("sex")["population"].transform("sum")
    filtered["share"] = (filtered["population"] / totals) * 100
    status_order = (
        filtered.groupby("status_display")["population"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    return (
        alt.Chart(filtered)
        .mark_bar()
        .encode(
            y=alt.Y("status_display:N", sort=status_order, title="Medeni durum"),
            x=alt.X("population:Q", title="Nüfus", axis=alt.Axis(format=",.0f")),
            color=alt.Color(
                "sex:N",
                scale=SEX_COLOR_SCALE,
                legend=SEX_COLOR_LEGEND,
            ),
            tooltip=[
                alt.Tooltip("status_display:N", title="Medeni durum"),
                alt.Tooltip("sex:N", title="Cinsiyet"),
                alt.Tooltip("population:Q", title="Nüfus", format=",.0f"),
                alt.Tooltip("share:Q", title="Pay (%)", format=".1f"),
            ],
        )
        .properties(height=max(240, 28 * len(status_order)))
    )


# -----------------------------------------------------------------------------
# Export helper
# -----------------------------------------------------------------------------


def build_export_workbook(
    population_ts: pd.DataFrame,
    settlement_df: pd.DataFrame,
    median_age_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    foreign_df: pd.DataFrame,
    household_df: pd.DataFrame,
    growth_df: pd.DataFrame,
    age_group_df: pd.DataFrame,
    marital_df: pd.DataFrame,
    single_age_df: pd.DataFrame,
) -> io.BytesIO:
    buffer = io.BytesIO()
    try:
        writer = pd.ExcelWriter(buffer, engine="xlsxwriter")
    except Exception:
        writer = pd.ExcelWriter(buffer, engine="openpyxl")
    with writer:
        population_ts.to_excel(writer, sheet_name="Nufus_Seri", index=False)
        settlement_df.to_excel(writer, sheet_name="Il_Ilce_Nufus", index=False)
        median_age_df.to_excel(writer, sheet_name="Ortanca_Yas", index=False)
        dependency_df.to_excel(writer, sheet_name="Yas_Bagimlilik", index=False)
        foreign_df.to_excel(writer, sheet_name="Yabanci_Nufus", index=False)
        household_df.to_excel(writer, sheet_name="Hanehalki", index=False)
        growth_df.to_excel(writer, sheet_name="Artis_Yogunluk", index=False)
        age_group_df.to_excel(writer, sheet_name="Yas_Grup", index=False)
        marital_df.to_excel(writer, sheet_name="Medeni_Durum", index=False)
        single_age_df.to_excel(writer, sheet_name="Tek_Yas", index=False)
    buffer.seek(0)
    return buffer


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


def main() -> None:
    st.title("TÜİK Nüfus ve Demografi Paneli")
    st.caption(
        "Adres bazlı nüfus kayıt sistemi (ADNKS) sonuçlarına dayanarak Türkiye ve İzmir için"
        " demografik göstergeleri cinsiyet kırılımlarıyla birlikte inceler."
    )

    try:
        population_ts = load_population_timeseries()
        settlement_df = load_population_settlement()
        median_age_df = load_median_age()
        dependency_df = load_age_dependency()
        foreign_df = load_foreign_population()
        household_df = load_household_size()
        growth_df = load_growth_density()
        age_group_df = load_age_group_population()
        marital_df = load_marital_status()
        single_age_df = load_single_age_population()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"Veriler yüklenirken bir hata oluştu: {exc}")
        return

    province_options = sorted(population_ts["province"].unique())
    default_provinces = ["İzmir", "Türkiye"]
    preselected = [p for p in default_provinces if p in province_options]
    if not preselected:
        preselected = province_options[:2]

    st.sidebar.header("Filtreler")
    selected_provinces = st.sidebar.multiselect(
        "Karşılaştırılacak iller",
        options=province_options,
        default=preselected,
    )
    if not selected_provinces:
        st.sidebar.warning("En az bir il seçmeniz gerekir.")
        return

    selected_province = st.sidebar.selectbox(
        "Detaylı inceleme için il seçin",
        options=province_options,
        index=province_options.index(preselected[0]) if preselected else 0,
    )

    latest_year = int(population_ts["year"].max())
    foreign_years = sorted(foreign_df["year"].unique())
    selected_foreign_year = st.sidebar.selectbox(
        "Yabancı nüfus yılı",
        options=foreign_years,
        index=foreign_years.index(max(foreign_years)),
    )

    age_years = sorted(age_group_df["year"].unique())
    selected_age_year = (
        st.sidebar.selectbox("Yaş dağılımı yılı", options=age_years, index=len(age_years) - 1)
        if age_years
        else latest_year
    )

    marital_years = sorted(marital_df["year"].unique())
    selected_marital_year = (
        st.sidebar.selectbox("Medeni durum yılı", options=marital_years, index=len(marital_years) - 1)
        if marital_years
        else latest_year
    )

    birth_years = sorted(single_age_df["year"].unique())
    selected_birth_year = (
        st.sidebar.selectbox("Doğum yılı (0 yaş)", options=birth_years, index=len(birth_years) - 1)
        if birth_years
        else latest_year
    )

    household_years = sorted(household_df["year"].unique())
    selected_household_year = st.sidebar.slider(
        "Hanehalkı karşılaştırma yılı",
        min_value=int(min(household_years)),
        max_value=int(max(household_years)),
        value=int(max(household_years)),
        step=1,
    )

    growth_periods = sorted({period for period in growth_df["period"].unique() if period})
    default_period = next((p for p in growth_periods if str(latest_year) in p), growth_periods[-1])

    st.markdown("### Nüfusun Zaman İçindeki Seyri")
    st.altair_chart(population_trend_chart(population_ts, selected_provinces), theme="streamlit", width="stretch")

    metrics_cols = st.columns(4)
    latest_population = (
        population_ts[population_ts["year"] == latest_year]
        .set_index("province")["population"]
    )
    if selected_province in latest_population.index:
        province_pop = latest_population[selected_province]
        metrics_cols[0].metric(
            f"{selected_province} nüfusu ({latest_year})",
            f"{province_pop:,.0f}",
        )
    turkey_pop = latest_population.get("Türkiye")
    if turkey_pop:
        metrics_cols[1].metric(
            f"Türkiye nüfusu ({latest_year})",
            f"{turkey_pop:,.0f}",
        )
    settlement_latest = settlement_df[
        (settlement_df["province"] == selected_province) & (settlement_df["area_type"] == "province")
    ]
    urban_total = settlement_latest["population_urban"].fillna(0).sum()
    rural_total = settlement_latest["population_rural"].fillna(0).sum()
    if urban_total or rural_total:
        share = urban_total / (urban_total + rural_total) if (urban_total + rural_total) else 0.0
        metrics_cols[2].metric(
            "Kentleşme oranı",
            f"%{share * 100:,.1f}",
            help="İl merkezi + ilçe merkezleri nüfusunun toplam nüfusa oranı.",
        )
    births_latest = single_age_df[
        (single_age_df["year"] == selected_birth_year)
        & (single_age_df["age"] == 0)
        & (single_age_df["sex"] == "Toplam")
    ]
    births_map = births_latest.set_index("province")["population"]
    if selected_province in births_map.index:
        metrics_cols[3].metric(
            f"{selected_province} 0 yaş nüfusu ({selected_birth_year})",
            f"{births_map[selected_province]:,.0f}",
        )

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown(f"### {selected_province} için Ortanca Yaş ve Cinsiyet")
        st.altair_chart(median_age_chart(median_age_df, selected_province), theme="streamlit", width="stretch")
    with col_right:
        st.markdown(f"### {selected_province} için Yaş Bağımlılık Oranları")
        st.altair_chart(dependency_chart(dependency_df, selected_province), theme="streamlit", width="stretch")

    st.markdown(f"### {selected_foreign_year} yilinda cinsiyete gore yabanci nufus")
    st.altair_chart(
        foreign_population_chart(foreign_df, selected_foreign_year),
        theme="streamlit",
        width="stretch",
    )

    st.markdown(f"### {selected_birth_year} yilinda 0 yas nufusu (yaklasik dogum sayisi)")
    st.altair_chart(
        newborn_chart(single_age_df, selected_birth_year),
        theme="streamlit",
        width="stretch",
    )

    pyramid_col, marital_col = st.columns(2)
    with pyramid_col:
        st.markdown(f"### {selected_province} yaş piramidi ({selected_age_year})")
        st.altair_chart(
            age_pyramid_chart(age_group_df, selected_province, selected_age_year),
            theme="streamlit",
            width="stretch",
        )
    with marital_col:
        st.markdown(f"### {selected_province} medeni durum profili ({selected_marital_year})")
        st.altair_chart(
            marital_status_chart(marital_df, selected_province, selected_marital_year),
            theme="streamlit",
            width="stretch",
        )

    st.markdown(f"### {selected_household_year} yılında en yüksek ortalama hanehalkı büyüklüğüne sahip iller")
    st.altair_chart(
        household_size_chart(household_df, selected_household_year),
        theme="streamlit",
        width="stretch",
    )

    density_col, growth_col = st.columns(2)
    with density_col:
        st.markdown(f"### {default_period} için nüfus yoğunluğu")
        st.altair_chart(
            growth_density_chart(growth_df, "population_density", default_period),
            theme="streamlit",
            width="stretch",
        )
    with growth_col:
        st.markdown(f"### {default_period} için yıllık nüfus artış hızı")
        st.altair_chart(
            growth_density_chart(growth_df, "annual_growth_rate", default_period),
            theme="streamlit",
            width="stretch",
        )

    growth_top_col, growth_bottom_col = st.columns(2)
    with growth_top_col:
        st.altair_chart(
            district_growth_leader_chart(settlement_df, descending=True),
            theme="streamlit",
            width="stretch",
        )
    with growth_bottom_col:
        st.altair_chart(
            district_growth_leader_chart(settlement_df, descending=False),
            theme="streamlit",
            width="stretch",
        )

    st.markdown("### İl/İlçe bazında nüfus dağılımı")
    province_filter = st.selectbox(
        "İlçe detayı görüntülenecek il",
        options=[p for p in province_options if p != "Türkiye"],
        index=province_options.index(selected_province) if selected_province != "Türkiye" else 0,
    )
    st.altair_chart(
        district_population_chart(settlement_df, province_filter),
        theme="streamlit",
        width="stretch",
    )
    district_table = settlement_df[
        (settlement_df["province"] == province_filter) & (settlement_df["area_type"] == "district")
    ].copy()
    district_table = district_table.sort_values("population_total", ascending=False)
    district_table.rename(
        columns={
            "area_name": "İlçe",
            "population_total": "Toplam nüfus",
            "population_urban": "İl/ilçe merkezi",
            "population_rural": "Belde/köy",
            "annual_growth_rate": "Yıllık artış hızı (‰)",
        },
        inplace=True,
    )
    st.dataframe(
        district_table.reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )

    export_buffer = build_export_workbook(
        population_ts,
        settlement_df,
        median_age_df,
        dependency_df,
        foreign_df,
        household_df,
        growth_df,
        age_group_df,
        marital_df,
        single_age_df,
    )
    st.download_button(
        "Verilerin Excel kopyasını indir",
        data=export_buffer.getvalue(),
        file_name="tuik_demografi_dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown(
        "_Not:_ Grafikler yalnızca TÜİK tarafından yayımlanan açık veri tablolarındaki bilgileri kullanır;"
        " cinsiyet kırılımı bulunmayan göstergeler toplam değerlerle gösterilmiştir."
    )


if __name__ == "__main__":
    main()
