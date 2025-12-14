import pandas as pd
import geopandas as gpd
import numpy as np
import os
import requests
import re
import xgboost as xgb
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from log import setup_log
from sklearn.metrics import mean_squared_error, r2_score
from functools import reduce

log = setup_log(log_dir="logs/")


class Mymodel:
    def __init__(self, data_dir, download_socioeco_data, years):
        self.data_dir = data_dir
        self.result_dir = self.data_dir + "/results/images/"
        self.region = None  # geographical region data
        self.ev_registration = None  # EV registration data
        # self.charging_station = None  # EV charging stations data
        self.new_ev_sale = None  # new EV sale
        self.vehicle_population = None  # vehicle population data
        self.fuel_prices = None
        self.se = {}  # socioeconomic data
        self.aggregated_data = pd.DataFrame()  # Aggregated data used for analysis
        self.multistep_data = None  # Multistep data used for multistep prediction
        self.feature_cols = None
        self.validation_results = None # Validation results
        self.download_socioeco_data = (
            download_socioeco_data  # Whether to download socioeconomic data
        )
        self.model = None
        self.years = years
        self.forecasting_results = None

    def read_data(self):
        # 1. California counties shapefile
        self.region = gpd.read_file("data/shapefile/CA_Counties_TIGER2016.shp")

        # 2. Ev registration and charging stations
        # California EV registration
        self.ev_registration = pd.read_csv(self.data_dir + "/CA_EV_registration.csv")
        # EV charging stations
        charging_stations_file = pd.read_excel(
            self.data_dir + "/EV_Chargers_Last_updated_03-20-2023_ada.xlsx",
            sheet_name=None,
        )
        self.charging_station = {}
        for quarter, df in charging_stations_file.items():
            self.charging_station[quarter] = df

        # New EV sales
        self.new_ev_sale = pd.read_excel(
            self.data_dir + "/New_ZEV_Sales_Last_updated_10-13-2025_ada.xlsx",
            sheet_name="County",
        )
        # self.new_ev_sale = self.new_ev_sale[self.new_ev_sale["FUEL_TYPE"] == "Electric"]
        self.new_ev_sale = (
            self.new_ev_sale.groupby(["Data Year", "COUNTY"])["Number of Vehicles"]
            .sum()
            .reset_index(name="New Vehicle Sales")
        )

        # Vehicle population
        self.vehicle_population = pd.read_excel(
            self.data_dir + "/Vehicle_Population_Last_updated_04-30-2025_ada.xlsx",
            sheet_name="County",
        )
        # self.vehicle_population = self.vehicle_population[
        #     self.vehicle_population["Fuel Type"] == "Battery Electric (BEV)"
        # ]
        self.vehicle_population = (
            self.vehicle_population.groupby(["Data Year", "County"])[
                "Number of Vehicles"
            ]
            .sum()
            .reset_index(name="Vehicle Population")
        )

        # fuel price data
        self.fuel_prices = pd.read_excel(
            self.data_dir + "/PET_PRI_GND_DCUS_SCA_A.xls", sheet_name="Data 1", header=2
        )
        self.fuel_prices["Date"] = self.fuel_prices["Date"].astype(int)

        # 3. Socioeconomic data per year
        # 3.1 Download socioeconomic data
        def _download_socioeconomic_data(download_dir: str = "data/socioeconomic/"):
            """
            This function download socioeconomic data to the specified directory.
            """

            if not self.download_socioeco_data:
                return

            base_url = "https://dof.ca.gov/media/docs/reports/demographic-reports/american-community-survey/"
            for year in self.years:
                if not os.path.exists(download_dir + str(year)):
                    os.makedirs(download_dir + str(year))
                file_names = [
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Pop-Race.xlsx",
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Inc-Pov-Emp.xlsx",
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_HealthIns.xlsx",
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Educ.xlsx",
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Social.xlsx",
                    f"Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Housing.xlsx",
                ]
                for file_name in file_names:
                    try:
                        response = requests.get(base_url + file_name)
                        response.raise_for_status()
                    except requests.exceptions.RequestException as e:
                        logging.warning(
                            f"Error downloading {file_name} for year {year}: {e}"
                        )
                        continue
                    with open(download_dir + str(year) + "/" + file_name, "wb") as f:
                        f.write(response.content)

        _download_socioeconomic_data(
            download_dir=self.data_dir + "/socioeconomic/",
        )

        # 3.2 Read socioeconomic data

        def _flatten_col(col):
            """
            Flatten multi-level column names.
            :param col:
            :return:
            """
            if not isinstance(col, tuple):
                chaos_string = [" - \n", " - \\n", " - ", "\n", "\\n"]
                for cs in chaos_string:
                    col = col.replace(cs, " ")
                col = " ".join(col.split())
                col = col.replace("  Estimtate", " Estimtate")
                return str(col)

            # Handle special columns for "Summary Level", "County", "Place"
            parts = []
            for s in col:
                if pd.isna(s) or str(s).startswith("Unnamed"):
                    continue
                parts.append(str(s))

            keyword = parts[-1]
            if keyword in ("Summary Level", "summary level", "sumlev", "Sumlev"):
                return "Summary Level"
            elif keyword in ("County", "county"):
                return "County"
            elif keyword in ("Place", "place"):
                return "Place"

            s = " ".join(parts) if parts else ""
            chaos_string = [" - \n", " - \\n", " - ", "\n", "\\n"]
            for cs in chaos_string:
                s = s.replace(cs, " ")

            # normalize spaces
            s = " ".join(s.split())
            s = s.replace("  Estimtate", " Estimtate")

            return s

        # population and median age
        for year in self.years:

            if year < 2018:
                header_pop = [4, 5]
                if year == 2014:
                    header_pop = [3, 4]
                header_edu = [4, 5]
                header_ins = [4, 5]
                header_housing = [3, 4]
                header_mort_value = 3
                header_house_cost = [3, 4, 5]
                header_income = [3, 4]
                header_social = [3, 4]
                header_employment = [3, 4, 5, 6]
            else:
                header_pop = 4
                header_edu = 4
                header_ins = 4
                header_housing = 4
                header_mort_value = 4
                header_house_cost = 4
                header_income = 4
                header_employment = 4
                header_social = 3

            self.se[year] = {}

            # population and median age
            self.se[year]["population_age"] = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Pop-Race.xlsx",
                sheet_name="Total Pop & Median Age",
                header=header_pop,
            )

            # income, poverty, and employment
            inc_pov = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Inc-Pov-Emp.xlsx",
                sheet_name=["Income", "Poverty"],
                header=header_income,
            )
            self.se[year]["income"] = inc_pov["Income"]
            # self.se[year]["poverty"] = inc_pov["Poverty"]
            self.se[year]["employment"] = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{str(year)}_{('0' + str(year-2004))[-2:]}_Inc-Pov-Emp.xlsx",
                sheet_name=["Employment Status"],
                header=header_employment,
            )["Employment Status"]

            # health insurance
            self.se[year]["health_insurance"] = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_HealthIns.xlsx",
                sheet_name="Health Insurance",
                header=header_ins,
            )

            # educational attainment
            self.se[year]["education"] = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Educ.xlsx",
                sheet_name="Educational Attainment",
                header=header_edu,
            )

            # social characteristics
            try:
                social_file = pd.read_excel(
                    self.data_dir
                    + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Social.xlsx",
                    sheet_name=[
                        "Households",
                        "Veterans & Disability",
                        "Nativity & Language",
                    ],
                    header=header_social,
                )
                self.se[year]["veterans_disability"] = social_file[
                    "Veterans & Disability"
                ]
            except ValueError:
                try:
                    social_file = pd.read_excel(
                        self.data_dir
                        + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Social.xlsx",
                        sheet_name=[
                            "Households",
                            "Veterans",
                            "Disability",
                            "Nativity & Language",
                        ],
                        header=header_social,
                    )
                    self.se[year]["veterans"] = social_file["Veterans"]
                    self.se[year]["disability"] = social_file["Disability"]
                except ValueError:
                    print(
                        f"Warning: Social characteristics data not found for year {year}."
                    )
                    continue

            self.se[year]["households"] = social_file["Households"]
            self.se[year]["nativity_language"] = social_file["Nativity & Language"]

            # Housing status
            housing_file = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Housing.xlsx",
                sheet_name=[
                    "Occupancy",
                    "Tenure",
                    "Units",
                ],
                header=header_housing,
            )
            housing_mort_value_file = pd.read_excel(
                self.data_dir
                + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Housing.xlsx",
                sheet_name=["Mortgage Status", "Value"],
                header=header_mort_value,
            )
            # try:
            #     housing_cost_file = pd.read_excel(
            #         self.data_dir
            #         + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Housing.xlsx",
            #         sheet_name=["Owner Costs", "Renter Costs"],
            #         header=header_house_cost,
            #     )
            #     self.se[year]["owner_costs"] = housing_cost_file["Owner Costs"]
            # except ValueError:
            #     housing_cost_file = pd.read_excel(
            #         self.data_dir
            #         + f"/socioeconomic/{year}/Web_ACS{year}_{('0' + str(year-2004))[-2:]}_Housing.xlsx",
            #         sheet_name=["Owner Costs ", "Renter Costs"],
            #         header=header_house_cost,
            #     )
            #     self.se[year]["owner_costs"] = housing_cost_file["Owner Costs "]
            self.se[year]["occupancy"] = housing_file["Occupancy"]
            self.se[year]["tenure"] = housing_file["Tenure"]
            self.se[year]["units"] = housing_file["Units"]
            self.se[year]["mortgage_status"] = housing_mort_value_file[
                "Mortgage Status"
            ]
            self.se[year]["value"] = housing_mort_value_file["Value"]
            # self.se[year]["renter_costs"] = housing_cost_file["Renter Costs"]

            # Normalize column
            for name, attribute in self.se[year].items():
                # Flatten column names
                if isinstance(attribute, dict):
                    print(name)
                    raise ValueError(
                        f"Socioeconomic data for {name} in year {year} is a dict, expected DataFrame."
                    )

                attribute.columns = [_flatten_col(c) for c in attribute.columns]

                # Keep county level data
                try:
                    mask = attribute["Summary Level"] == 50
                except KeyError:
                    print(
                        f"Warning: 'Summary Level' column not found in {name} for year {year}. name is {attribute.columns}."
                    )
                    continue
                attribute = attribute[mask].reset_index(drop=True)

                # Keep necessary columns only
                kept_columns = ["Geography"] + [
                    c
                    for c in attribute.columns
                    if re.search(r"Percent$|Estimate$", c)
                    and not re.search(r"Error", c)
                ]

                self.se[year][name] = attribute[kept_columns].copy()

    def manually_select_features(self):

        # Employment
        for year in self.years:
            if year < 2018:
                self.se[year]["employment"] = self.se[year]["employment"].iloc[:, :-3]

            else:
                self.se[year]["employment"].columns = ["Geography"] + [
                    c.replace("Population 16 years and over ", "")
                    for c in self.se[year]["employment"].columns
                    if re.search("Population 16 years and over ", c)
                ]

    def aggregate_data(self):
        # Aggregate EV registration by county and year
        self.ev_registration = self.ev_registration[
            self.ev_registration["County GEOID"] != "Unknown"
        ]
        self.ev_registration["County GEOID"] = self.ev_registration[
            "County GEOID"
        ].astype(int)
        self.ev_registration = (
            self.ev_registration.groupby(["County GEOID", "Registration Valid Date"])
            .size()
            .rename("EV Count")
            .reset_index()
        )
        self.ev_registration["Registration Valid Date"] = (
            self.ev_registration["Registration Valid Date"]
            .apply(lambda x: x.split("-")[0])
            .astype(int)
        )
        self.ev_registration.rename(
            columns={"Registration Valid Date": "year"}, inplace=True
        )

        # Prepare region data
        self.region.to_crs(epsg=3310, inplace=True)  # California Albers
        self.region["area"] = self.region.geometry.area
        self.region = self.region[["GEOID", "NAME", "area", "geometry"]]
        self.region["GEOID"] = self.region["GEOID"].astype(int)
        region = pd.DataFrame(self.region.drop(columns=["geometry"]))

        # Merge EV registration with region data
        self.ev_registration = self.ev_registration.merge(
            region, left_on="County GEOID", right_on="GEOID", how="left"
        ).drop(columns=["County GEOID"])

        ## Aggregate vehicle sale and population data
        self.ev_registration = self.ev_registration.merge(
            self.new_ev_sale,
            left_on=["NAME", "year"],
            right_on=["COUNTY", "Data Year"],
            how="left",
        ).drop(columns=["COUNTY", "Data Year"])

        self.ev_registration = self.ev_registration.merge(
            self.vehicle_population,
            left_on=["NAME", "year"],
            right_on=["County", "Data Year"],
            how="left",
        ).drop(columns=["County", "Data Year"])

        # Merge socioeconomic data
        for year, se_data in self.se.items():
            year_i = reduce(
                lambda left, right: pd.merge(left, right, on=["Geography"]),
                se_data.values(),
            )
            # Add year column
            year_i.loc[:, "year"] = year
            self.aggregated_data = pd.concat(
                [year_i, self.aggregated_data], axis=0, ignore_index=True
            )
        self.aggregated_data["Geography"] = self.aggregated_data[
            "Geography"
        ].str.replace(" County", "", regex=False)
        self.aggregated_data = self.aggregated_data.merge(
            self.ev_registration,
            left_on=["Geography", "year"],
            right_on=["NAME", "year"],
            how="left",
        )

        # merge fuel price
        self.aggregated_data = self.aggregated_data.merge(
            self.fuel_prices, left_on="year", right_on="Date", how="left"
        )

        self.aggregated_data.drop(columns=["Total housing units Estimate_y"])
        self.aggregated_data.rename(
            columns={"Total housing units Estimate_x": "Total Housing Units Estimate"},
            inplace=True,
        )

        self.aggregated_data.drop(columns=["NAME"], inplace=True)
        self.aggregated_data["EV Count"].fillna(0, inplace=True)
        self.aggregated_data["New Vehicle Sales"] = (
            self.ev_registration["New Vehicle Sales"].fillna(0).astype(int)
        )
        self.aggregated_data["Vehicle Population"] = (
            self.ev_registration["Vehicle Population"].fillna(0).astype(int)
        )

        # Calculate vehicle rate
        self.aggregated_data["Vehicle Population Rate"] = (
            self.aggregated_data["Vehicle Population"]
            / self.aggregated_data["Total Population Estimate"]
            * 10000
        )
        self.aggregated_data["New Vehicle Sales Rate"] = (
            self.aggregated_data["New Vehicle Sales"]
            / self.aggregated_data["Total Population Estimate"]
            * 10000
        )

        self.aggregated_data.dropna(axis=1, inplace=True)

    def data_histogram(self):
        df = self.aggregated_data

        # Create a figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        # Plot 1: Raw EV Adoption Rate
        sns.histplot(
            df["EV Count"] / df["Total Population Estimate"] * 10000,
            kde=True,
            color="blue",
            bins=50,
            ax=axes[0],  # Assign to the first subplot
        )
        axes[0].set_title("(a) Distribution of EV Adoption Rate", fontsize=20, y=-0.1)
        axes[0].set_xlabel("Adoption Rate (EVs per 10000 people)", fontsize=16)
        axes[0].set_ylabel("Frequency (Count of County-Years)", fontsize=16)

        # Plot 2: Log-Transformed EV Adoption Rate
        ev_log = np.log1p(df["EV Count"] / df["Total Population Estimate"] * 10000)
        sns.histplot(
            ev_log,
            kde=True,
            color="green",
            bins=50,
            ax=axes[1],  # Assign to the second subplot
        )
        axes[1].set_title("(b) Distribution of Log-Transformed EV Adoption Rate", fontsize=20, y=-0.1)
        axes[1].set_xlabel("Log Adoption Rate (log(EVs per 10000 people + 1))", fontsize=16)
        axes[1].set_ylabel("Frequency (Count of County-Years)", fontsize=16)

        # Adjust layout to prevent overlap and save
        plt.tight_layout()
        plt.savefig(self.result_dir + "Combined_EV_Adoption_Rate_Distribution.png")

    def spatial_plot(self):
        """
        Plots the spatial distribution of EV adoption rates for the latest two years.
        """
        df = self.aggregated_data.copy()
        latest_year = df["year"].max()
        start_year = latest_year - 5

        df_latest = df[df["year"] == latest_year]
        df_start = df[df["year"] == start_year]

        # Helper to merge and calculate rate
        def prepare_gdf(data_subset):
            gdf = self.region.merge(
                data_subset, left_on="NAME", right_on="Geography", how="left"
            )
            gdf["adoption_rate"] = gdf.apply(
                lambda x: (
                    (x["EV Count"] / x["Total Population Estimate"] * 10000)
                    if x["Total Population Estimate"] > 0
                    else 0
                ),
                axis=1,
            )
            return gdf

        gdf_latest = prepare_gdf(df_latest)
        gdf_prev = prepare_gdf(df_start)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        gdf_prev.plot(
            column="adoption_rate",
            ax=axes[0],
            legend=True,
            cmap="OrRd",
            vmin=0,
            vmax=150,
            missing_kwds={"color": "lightgrey"},
            legend_kwds={"label": "Adoption Rate (per 10k people)"},
        )
        axes[1].get_figure().axes[-1].yaxis.label.set_size(16)
        axes[0].set_title(f"(a) EV Adoption Rate by County in {start_year}", fontsize=20, y=-0.1)
        axes[0].axis("off")

        gdf_latest.plot(
            column="adoption_rate",
            ax=axes[1],
            legend=True,
            cmap="OrRd",
            vmin=0,
            vmax=150,
            missing_kwds={"color": "lightgrey"},
            legend_kwds={"label": "Adoption Rate (per 10k people)"},
        )
        axes[1].set_title(f"(b) EV Adoption Rate by County in {latest_year}", fontsize=20, y=-0.1)
        axes[1].axis("off")
        axes[1].get_figure().axes[-1].yaxis.label.set_size(16)
        plt.tight_layout()
        plt.savefig(self.result_dir + "Spatial_EV_Adoption_Rate_Comparison.png",bbox_inches="tight")

    def spatialtemporal_feature(self, time_lag=2):
        """
        Calculates Target Spatial Lags and Temporal Lags.
        """
        if self.aggregated_data.empty:
            raise ValueError("Aggregated data is empty. Run aggregate_data() first.")

        df = self.aggregated_data.copy()

        pop_col = "Total Population Estimate"

        # 1. Calculate adoption rate (EV count/ ten thousand people)
        df["adoption_rate"] = df.apply(
            lambda x: (x["EV Count"] / x[pop_col] * 10000) if x[pop_col] > 0 else 0,
            axis=1,
        )

        # 2. Generate spatial lag features
        gdf_lookup = self.region[["NAME", "geometry"]].drop_duplicates()
        gdf_lookup["neighbors"] = None

        for index, row in gdf_lookup.iterrows():
            neighbors = gdf_lookup[gdf_lookup.geometry.touches(row["geometry"])]
            gdf_lookup.at[index, "neighbors"] = neighbors["NAME"].tolist()
        neighbor_map = dict(zip(gdf_lookup["NAME"], gdf_lookup["neighbors"]))

        def get_spatial_lag(row, full_df):
            neighbors = neighbor_map.get(row["Geography"], [])
            if not neighbors:
                return 0
            prev_year = row["year"] - 1
            neighbor_data = full_df[
                (full_df["Geography"].isin(neighbors)) & (full_df["year"] == prev_year)
            ]
            if neighbor_data.empty:
                return 0
            return neighbor_data["adoption_rate"].mean()

        spatial_lags = []

        df = df.sort_values(by=["year", "Geography"])

        for idx, row in df.iterrows():
            lag = get_spatial_lag(row, df)
            spatial_lags.append({"index": idx, "spatial_lag": lag})

        lag_df = pd.DataFrame(spatial_lags).set_index("index")
        df = df.join(lag_df)
        df["spatial_lag"] = df["spatial_lag"].fillna(0)

        # 3. Time lag Feature
        df = df.sort_values(by=["Geography", "year"])
        time_lags = np.arange(1, time_lag + 1)
        for lag in time_lags:
            df[f"adoption_rate_time_lag_{lag}"] = (
                df.groupby("Geography")["adoption_rate"].shift(lag).fillna(0)
            )
            # df[f"New Vehicle Sales_lag_{lag}"] = (
            #     df.groupby("Geography")["New Vehicle Sales"].shift(lag).fillna(0)
            # )
            # df[f"Vehicle Population_lag_{lag}"] = (
            #     df.groupby("Geography")["Vehicle Population"].shift(lag).fillna(0)
            # )

        self.aggregated_data = df
        logging.info(f"Spatial-temporal feature Generated, column shape: {self.aggregated_data.shape}")

    def generate_multistep_data(self, n_steps=2):
        """
        Prepares data for multi-step forecasting.
        """
        logging.info(f"Generating Multi-Step Data for {n_steps} steps...")

        if self.aggregated_data is None:
            raise ValueError("Run spatialtemporal_feature() first.")

        df = self.aggregated_data.copy()
        df = df.sort_values(by=["Geography", "year"])

        expanded_data = []

        for geography, group in df.groupby("Geography"):
            group = group.sort_values("year")

            for step in range(1, n_steps + 1):
                step_df = group.copy()
                step_df["forecast_step"] = step
                step_df["target_future"] = group["adoption_rate"].shift(-step)
                step_df["target_year"] = step_df["year"] + step

                expanded_data.append(step_df)

        final_df = pd.concat(expanded_data)

        # Drop rows where target is NaN
        final_df = final_df.dropna(subset=["target_future"])

        self.multistep_data = final_df
        logging.info(f"Multi-step data generated. Shape: {self.multistep_data.shape}")

    def xgb_train(self):
        """
        Train and validate XGBoost model for EV adoption forecasting.
        :return:
        """
        logging.info("Starting XGBoost Training & Validation...")

        df = self.multistep_data.copy()

        # 1. exclude non-feature columns
        exclude_cols = [
            "EV Count",
            "Total Population Estimate",
            "adoption_rate",
            "target_future",
            "target_log",
            "geometry",
            "year",
            "target_year",
            "Geography",
            "forecast_step",
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        feature_cols = [
            c for c in feature_cols if df[c].dtype in ["float64", "int64", "int32"]
        ]

        self.feature_cols = feature_cols

        # 2. adoption rate distribution is right skewed, apply log-transform
        df["target_log"] = np.log1p(df["target_future"])

        # 3. Split train and test data
        max_ground_truth_year = df["target_year"].max()
        train_df = df[df["target_year"] < max_ground_truth_year]
        test_df = df[df["target_year"] == max_ground_truth_year]

        X_train = train_df[feature_cols]
        y_train = train_df["target_log"]
        X_test = test_df[feature_cols]
        y_test = test_df["target_log"]

        # 4. Train
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            n_jobs=-1,
            random_state=42,
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        # 5. Validate Metrics
        predictions = np.expm1(self.model.predict(X_test))
        actuals = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        logging.info(f"VALIDATION RESULTS (Year {max_ground_truth_year})")
        logging.info(f"RMSE: {rmse:.4f}")
        logging.info(f"R2 Score: {r2:.4f}")

        # 6. Save Results
        self.validation_results = test_df[["Geography", "year"]].copy()
        self.validation_results["actual"] = actuals
        self.validation_results["predicted"] = predictions
        self.validation_results["residual"] = actuals - predictions

        # Performs SHAP analysis and generates plots.
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_test)

        # Plot Beeswarm
        plt.figure(figsize=(12, 18))
        shap.plots.beeswarm(shap_values, max_display=16, show=False)
        plt.xlabel('SHAP value: Impact on Model Output (Log-Odds of Adoption)', fontsize=16)
        plt.savefig(self.result_dir + "shap_beeswarm.png", bbox_inches="tight")

        plt.figure(figsize=(12, 18))
        shap.plots.scatter(shap_values[:, "adoption_rate_time_lag_1"], color=shap_values, show=False)
        plt.savefig(self.result_dir + "shap_scatter.png", bbox_inches="tight")

        return self.validation_results

    def xgb_forecast(self, steps=2):
        """
        Generate future forecasts for the future `steps` years.
        """
        logging.info(f"Generating Future Forecasts for next {steps} years...")

        max_year = self.aggregated_data["year"].max()
        base_df = self.aggregated_data[self.aggregated_data["year"] == max_year].copy()
        base_df = base_df.sort_values("Geography").reset_index(drop=True)

        # 1. Spatial adjacency index mapping

        geo_to_idx = {name: idx for idx, name in enumerate(base_df["Geography"])}
        gdf_lookup = self.region.set_index("NAME")["geometry"]

        adj_indices = []
        for current_geo in base_df["Geography"]:
            if current_geo not in gdf_lookup.index:
                adj_indices.append([])  # Handle missing geometries gracefully
                continue

            current_geom = gdf_lookup[current_geo]
            neighbors = gdf_lookup[gdf_lookup.touches(current_geom)].index.tolist()
            neighbor_idxs = [geo_to_idx[n] for n in neighbors if n in geo_to_idx]
            adj_indices.append(neighbor_idxs)

        # 2. Recursive Forcast
        future_predictions = []

        target_current = base_df["adoption_rate"].values
        lag_1_current = target_current
        lag_2_current = base_df[
            "adoption_rate_time_lag_1"
        ].values  # This represents 2019 data

        for step in range(1, steps + 1):
            step_df = base_df.copy()

            # Update temporal metadata
            step_df["forecast_step"] = step
            step_df["target_year"] = max_year + step

            # 1. Update Temporal Lags
            step_df["adoption_rate_time_lag_2"] = lag_2_current
            step_df["adoption_rate_time_lag_1"] = lag_1_current

            # 2. Update Spatial Lag
            new_spatial_lags = []
            for n_idxs in adj_indices:
                if n_idxs:
                    # Average the rates of the neighbors
                    val = np.mean(lag_1_current[n_idxs])
                else:
                    val = 0
                new_spatial_lags.append(val)

            step_df["spatial_lag"] = new_spatial_lags

            # Predict
            X_future = step_df[self.feature_cols]
            pred_log = self.model.predict(X_future)
            pred_vals = np.expm1(pred_log)

            # Next recursion
            lag_2_current = lag_1_current
            lag_1_current = pred_vals

            # Store Results
            result = step_df[["Geography", "year"]].copy()
            result["forecast_year"] = step_df["target_year"]
            result["predicted_adoption_rate"] = pred_vals
            future_predictions.append(result)

        self.forecasting_results = pd.concat(future_predictions)
        logging.info(f"Forecasting future {steps} years completed.")

    def plot_error_distribution(self):
        """
        Plots the RMSE distribution of residuals.
        """
        if not hasattr(self, "validation_results"):
            print("No validation results to plot.")
            return

        df = self.validation_results
        rmse = np.sqrt(mean_squared_error(df["actual"], df["predicted"]))

        plt.figure(figsize=(14, 6))

        # Plot 1: Histogram of Residuals
        plt.subplot(1, 2, 1)
        sns.histplot(df["residual"], kde=True, color="crimson", bins=30)
        plt.axvline(x=0, color="black", linestyle="--")
        plt.title(f"(a) Residual Distribution (RMSE: {rmse:.2f})", fontsize=20, y=-0.2)
        plt.xlabel("Prediction Error (Actual - Predicted)", fontsize=16)
        plt.ylabel("Count", fontsize=16)
        logging.info("Generating RMSE distribution plot...")

        # Plot 2: Actual vs Predicted Scatter
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=df["actual"], y=df["predicted"], alpha=0.6)

        # Add a perfect prediction line (y=x)
        max_val = max(df["actual"].max(), df["predicted"].max())
        plt.plot([0, max_val], [0, max_val], "k--", lw=2)

        plt.title("(b) Actual vs Predicted Adoption Rate", fontsize=20, y=-0.2)
        plt.xlabel("Actual EV adoption Rate", fontsize=16)
        plt.ylabel("Predicted EV adoption Rate", fontsize=16)
        logging.info("Generating Actual vs Predicted Adoption Rate...")

        plt.tight_layout()
        plt.savefig(self.result_dir + "Residuals_and_Actual_vs_Predicted.png")

    def plot_feature_importance(self, top_n=10):
        """
        Plots the Top N most important features based on Information Gain.
        """
        if self.model is None:
            print("Model not trained yet.")
            return

        # 1. Get Importance Score (Gain)
        importance = self.model.get_booster().get_score(importance_type="gain")

        # 2. Convert to DataFrame
        importance_df = (
            pd.DataFrame(
                {
                    "feature": list(importance.keys()),
                    "importance": list(importance.values()),
                }
            )
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        # 3. Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance_df,
            x="importance",
            y="feature",
            palette="viridis",
            hue="feature",
            legend=False,
        )

        plt.title(f"Top {top_n} Features by Importance (Gain)", fontsize=20)
        plt.xlabel("Average Gain (Improvement to Model Accuracy)", fontsize=16)
        plt.ylabel("Feature Name", fontsize=18)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(self.result_dir + "Feature_importance.png", bbox_inches="tight")


if __name__ == "__main__":
    ca = Mymodel("data", download_socioeco_data=False, years=list(range(2012, 2021)))
    ca.read_data()
    ca.manually_select_features()
    ca.aggregate_data()
    ca.data_histogram()
    ca.spatial_plot()
    ca.spatialtemporal_feature(time_lag=2)
    ca.generate_multistep_data(n_steps=2)
    ca.xgb_train()
    ca.xgb_forecast(steps=2)
    ca.plot_error_distribution()
    ca.plot_feature_importance()
