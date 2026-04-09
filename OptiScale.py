# import libraries
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import pwlf
from pathlib import Path
import re
from time import perf_counter


EARTH_RADIUS = 6371

FIG_DPI = 300

# Function to obtain the divisors of a number, with 2 decimal places
def divisors(target, precision=2):
    """
    Find all divisors of a number up to the given decimal precision.

    Args:
        target (float): The number whose divisors to find.
        precision (int): Number of decimal places to consider (default 2 for hundredths).

    Returns:
        list[float]: List of divisors rounded to the given precision.
    """
    scale = 10**precision
    scaled_target = int(round(target * scale))
    divisors = []

    for i in range(1, scaled_target + 1):
        if scaled_target % i == 0:  # integer division check (exact)
            divisors.append(round(i / scale, precision))

    return divisors


# Function to generate all grids with the given lat and lon sizes, without years and with area in km2
def generate_all_grids(xmin, ymin, xmax, ymax, lon_size, lat_size, size_counts=None):
    if size_counts is None:
        size_counts = {}  # intialize if not passed

    lon_grid = np.round(
        np.arange(xmin, xmax, lon_size), 2
    )  # its important to round the values to avoid floating point errors
    lat_grid = np.round(np.arange(ymin, ymax, lat_size), 2)

    size = int(((ymax - ymin) / lat_size) * ((xmax - xmin) / lon_size))

    if size not in size_counts:
        size_counts[size] = 0
    size_counts[size] += 1
    size_label = f"{size}{chr(96 + size_counts[size])}"  # unique label for each size

    all_grids = []
    for lon in lon_grid:
        for lat in lat_grid:
            # Calculate area using the average latitude for the cell, elipsoidal approximation
            cell_width = (
                (lon_size / 360)
                * 2
                * np.pi
                * EARTH_RADIUS
                * np.cos(np.radians(lat + lat_size / 2))
            )  # Calculate the width of the cell
            cell_height = (
                (lat_size / 360) * 2 * np.pi * EARTH_RADIUS
            )  # Calculate the height of the cell
            area_km2 = cell_width * cell_height  # Calculate the area of the cell in km2

            # Append each cell to the list
            all_grids.append(
                {
                    "lat_size": lat_size,
                    "lon_size": lon_size,
                    "lon_min": lon,
                    "lat_min": lat,
                    "lon_max": np.round(lon + lon_size, 2),
                    "lat_max": np.round(lat + lat_size, 2),
                    "area_km2": round(area_km2),
                    "size": size_label,
                }
            )

    return pd.DataFrame(all_grids), size_counts


# Function to assign the grid cell (generate_all_grids) to each point in the dataframe (df)
def assign_to_grid(row, grid_df, assigned_points=None):
    if grid_df is None or grid_df.empty:
        return None

    if assigned_points is None:
        assigned_points = {}

    point_key = row.name  # Using the index of the point as a unique key

    # Check if the point has already been assigned a grid cell
    if point_key in assigned_points:
        return assigned_points[
            point_key
        ]  # If the point has already been assigned, return the assigned cell

    # Find the grid cell that matches the longitude and latitude of the point, including the edges
    grid = grid_df[
        (row["lon"] >= grid_df["lon_min"])
        & (row["lon"] <= grid_df["lon_max"])
        & (row["lat"] >= grid_df["lat_min"])
        & (row["lat"] <= grid_df["lat_max"])
    ]

    # If a matching grid cell is found, assign its value to the point's grid_cell attribute
    if not grid.empty:
        assigned_cell_id = grid.index[0]
        assigned_points[point_key] = assigned_cell_id
        return assigned_cell_id

    return None


# Function to calculate diversity indices per spatial resolution
def diversity_indices(df):
    total_individuals = df.groupby("grid_cell").size().reset_index(name="abundance")
    individuals_per_species = df.groupby(["grid_cell", "species"]).size().reset_index(name="individuals_per_species").merge(total_individuals, on="grid_cell", how="left")
    individuals_per_species["proportion"] = individuals_per_species["individuals_per_species"] / individuals_per_species["abundance"]

    richness = df.groupby("grid_cell")["species"].nunique().reset_index(name="richness")
    shannon_index = individuals_per_species.groupby("grid_cell").apply(lambda g: -np.sum(g["proportion"] * np.log(g["proportion"])), include_groups=False).reset_index(name="shannon")
    simpson_index = individuals_per_species.groupby("grid_cell").apply(lambda g: 1 - np.sum(g["proportion"] ** 2), include_groups=False).reset_index(name="simpson")

    diversity_df = total_individuals.merge(richness, on="grid_cell", how="left")
    diversity_df["margalef"] = ((diversity_df["richness"] - 1) / np.log(diversity_df["abundance"].replace(0, np.nan))).round(4).fillna(0)
    diversity_df["menhinick"] = (diversity_df["richness"] / np.sqrt(diversity_df["abundance"].replace(0, np.nan))).round(4).fillna(0)
    diversity_df = diversity_df.merge(shannon_index, on="grid_cell", how="left")
    diversity_df = diversity_df.merge(simpson_index, on="grid_cell", how="left")

    return diversity_df


# Function to calculate statistics for the final DataFrame
def statistics(final_df):
    stats_df = (
        final_df.groupby(["size", "lat_size", "lon_size"])
        .agg(
            median_r=("richness", "median"),
            mean_r=("richness", "mean"),
            median_ab=("abundance", "median"),
            mean_ab=("abundance", "mean"),
            median_S=("shannon", "median"),
            median_Sp=("simpson", "median"),
            median_Mg=("margalef", "median"),
            median_Mn=("menhinick", "median"),
            mean_S=("shannon", "mean"),
            mean_Sp=("simpson", "mean"),
            mean_Mg=("margalef", "mean"),
            mean_Mn=("menhinick", "mean"),
            means_area_km2=("area_km2", "mean"),
            variance_r=("richness", "var"),
            variance_ab=("abundance", "var"),
            variance_S=("shannon", "var"),
            variance_Sp=("simpson", "var"),
            variance_Mg=("margalef", "var"),
            variance_Mn=("menhinick", "var"),
            N=("size", "count"),
        )
        .reset_index()
    )
    return stats_df


# Function to select the closest resolutions to the second breakpoint
def selected_resolutions(stats_df, second_breakpoint, n, select):

    stats_df["distance_to_sb"] = abs(stats_df["means_area_km2"] - second_breakpoint)

    if select == "below":
        closest_resolutions = stats_df[
            stats_df["means_area_km2"] <= second_breakpoint
        ].nsmallest(n, "distance_to_sb")
    elif select == "above":
        closest_resolutions = stats_df[
            stats_df["means_area_km2"] >= second_breakpoint
        ].nsmallest(n, "distance_to_sb")
    else:
        closest_resolutions = stats_df.nsmallest(n, "distance_to_sb")
    return closest_resolutions


# Function to calculate the coefficient of variation (CV) for species richness of years*spatial interaction = Spatial-temporal CV
def calculate_cv(df):
    if df.mean() == 0:
        return 0
    return df.std() / df.mean()


# Function to generate all grids with the given lat and lon sizes, with years and with area in km2 for time windows (TW) and analysis
def generate_period_grids(xmin, ymin, xmax, ymax, lat_sizes, lon_sizes, periods, size_labels_map=None):
    if size_labels_map is None:
        size_labels_map = {}

    all_grids = []
    for lat_size in lat_sizes:
        for lon_size in lon_sizes:
            key = (lat_size, lon_size)
            lon_grid = np.round(np.arange(xmin, xmax, lon_size), 2)
            lat_grid = np.round(np.arange(ymin, ymax, lat_size), 2)
            size = int(((ymax - ymin) / lat_size) * ((xmax - xmin) / lon_size))
            size_label = size_labels_map.get(
                key, f"{size}{chr(97)}"
            )  # fallback if not in map

            for lon in lon_grid:
                for lat in lat_grid:
                    cell_width = (
                        (lon_size / 360)
                        * 2
                        * np.pi
                        * EARTH_RADIUS
                        * np.cos(np.radians(lat + lat_size / 2))
                    )
                    cell_height = (lat_size / 360) * 2 * np.pi * EARTH_RADIUS
                    area_km2 = cell_width * cell_height

                    for period in periods:
                        all_grids.append(
                            {
                                "lat_size": lat_size,
                                "lon_size": lon_size,
                                "lon_min": lon,
                                "lat_min": lat,
                                "lon_max": np.round(lon + lon_size, 2),
                                "lat_max": np.round(lat + lat_size, 2),
                                "area_km2": round(area_km2),
                                "size": size_label,
                                "year": period,
                            }
                        )

    grid_df = pd.DataFrame(all_grids)
    unique_combinations = (
        grid_df[["lon_min", "lat_min", "lon_max", "lat_max", "area_km2"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_combinations["grid_cell"] = unique_combinations.index
    grid_df = grid_df.merge(
        unique_combinations,
        on=["lon_min", "lat_min", "lon_max", "lat_max", "area_km2"],
        how="left",
    )
    return grid_df, size_labels_map


# Function to assign the grid cell (generate_period_grids) to each point in the time-series dataframe (df) 
def assign_to_grid_p(row, grid_df, assigned_points):
    if grid_df is None or grid_df.empty:
        return None
    if (row.name, row["period"]) in assigned_points:
        return assigned_points[(row.name, row["period"])]
    grid = grid_df[
        (row["lon"] >= grid_df["lon_min"])
        & (row["lon"] <= grid_df["lon_max"])
        & (row["lat"] >= grid_df["lat_min"])
        & (row["lat"] <= grid_df["lat_max"])
    ]
    if not grid.empty:
        assigned_grid_cell = grid["grid_cell"].iloc[0]
        assigned_points[(row.name, row["period"])] = assigned_grid_cell
        return assigned_grid_cell
    return None


# Function to calculate diversity indices for slected resolutions and time windows combinations, with years and with area in km2 for time windows (TW) and analysis
def calculate_diversity_indices_p(df):
    total_individuals = (df.groupby(["grid_cell", "period"]).size().reset_index(name="abundance"))
    individuals_per_species = (df.groupby(["grid_cell", "period", "species"]).size().reset_index(name="individuals_per_species"))
    individuals_per_species = individuals_per_species.merge(total_individuals, on=["grid_cell", "period"])
    individuals_per_species["proportion"] = (individuals_per_species["individuals_per_species"]/ individuals_per_species["abundance"])
    richness = (df.groupby(["grid_cell", "period"])["species"].nunique().reset_index(name="richness"))
    shannon_index = (individuals_per_species.groupby(["grid_cell", "period"]).apply(lambda group: -np.sum(group["proportion"] * np.log(group["proportion"])), include_groups=False).reset_index(name="shannon"))
    simpson_index = (individuals_per_species.groupby(["grid_cell", "period"]).apply(lambda group: 1 - np.sum(group["proportion"] ** 2), include_groups=False).reset_index(name="simpson"))
    diversity_df = total_individuals.merge(richness, on=["grid_cell", "period"])
    diversity_df["log_abundance"] = np.log(diversity_df["abundance"].replace(0, np.nan))
    diversity_df["margalef"] = (((diversity_df["richness"] - 1) / diversity_df["log_abundance"]).round(4).fillna(0))
    diversity_df["menhinick"] = ((diversity_df["richness"]/ np.sqrt(diversity_df["abundance"].replace(0, np.nan))).round(4).fillna(0))
    diversity_df = diversity_df.merge(shannon_index, on=["grid_cell", "period"], how="left")
    diversity_df = diversity_df.merge(simpson_index, on=["grid_cell", "period"], how="left")
    return diversity_df


# Function to assign colors to each TW in the porcess control charts
def color_for_period(time_windows):
    fixed_colors = {1: "blue", 5: "green", 10: "orange"}

    input_intervals = tuple(time_windows)
    if input_intervals == (1, 5, 10):
        return fixed_colors
    else:
        chosen_colormap = "Accent" # You can choose any colormap you like from Matplotlib
        cmap = plt.cm.get_cmap(chosen_colormap, len(time_windows)) # Get the colormap

        random_colors = {}

        for i, interval in enumerate(time_windows):
            rgb_color = cmap(i) # Obtain the RGB color from the colormap
            hex_color = mcolors.rgb2hex(rgb_color) # Convert RGB to HEX format
            random_colors[interval] = hex_color # Assign the HEX color to the interval

        return random_colors


# Funtion to select optimal resolutions (OpR) or SR-TW combinations based on outliers_count, mean_cv_richness, diff_cv_richness from "Process Control Chart Analysis" approach
def select_optimal_combinations(df):
    df = df.copy()
    mean_outliers_count = df["outliers_count"].mean()
    mean_mean_cv_richness = df["mean_cv_richness"].mean()
    mean_diff_cv_richness = df["diff_cv_richness"].mean()

    # Step 1: prioritize lower outliers_count
    c1 = df[df["outliers_count"] <= mean_outliers_count]
    if c1.empty:
        min_out = df["outliers_count"].min()  # find the minimum outliers_count
        res = df[df["outliers_count"] == min_out].copy()
    else:
        if len(c1) == 1:
            res = c1.copy()
        else:
            # Step 2: mean_cv_richness only when c1 > 1
            c2 = c1[c1["mean_cv_richness"] <= mean_mean_cv_richness]
            if c2.empty:
                min_cv = c1[
                    "mean_cv_richness"
                ].min()  # find the minimum mean_cv_richness
                c2 = c1[c1["mean_cv_richness"] == min_cv].copy()

            if len(c2) == 1:
                res = c2.copy()
            else:
                # Step 3: diff_cv_richness only when c2 > 1
                c3 = c2[c2["diff_cv_richness"] <= mean_diff_cv_richness]
                if c3.empty:
                    min_diff = c2[
                        "diff_cv_richness"
                    ].min()  # find the minimum diff_cv_richness
                    c3 = c2[c2["diff_cv_richness"] == min_diff].copy()
                res = c3.copy()
                
                cols = ["outliers_count", "mean_cv_richness", "diff_cv_richness"]

                if len(res) > 1:  # multiplicative index as last resort
                    tmp = res.copy()
                    # normalization min-max for each column
                    for col in cols:
                        min_val = tmp[col].min()
                        max_val = tmp[col].max()
                        if max_val - min_val == 0:
                            tmp[col] = 0
                        else:
                            tmp[col] = (tmp[col] - min_val) / (max_val - min_val)
                    tmp["multiplicative_index"] = (
                        tmp["mean_cv_richness"]
                        * tmp["diff_cv_richness"]
                        * tmp["outliers_count"]
                    )
                    # Select the row(s) with the minimum multiplicative index
                    min_index_val = tmp["multiplicative_index"].min()
                    res = tmp[tmp["multiplicative_index"] == min_index_val].copy()

    res = res.sort_values(["outliers_count", "mean_cv_richness", "diff_cv_richness"])
    return res


def main():
    """
    Run the script:

    python <path to script>/OptiScale.py <path to data file>/<data>.csv xmin=<int> xmax=<int> ymin=<int> ymax=<int> n=<int> s=<below|above|none> TW=<(x,y,z,...)>
    example: python OptiScale.py Data_Ebro_1950-2022.csv xmin=-2 xmax=0 ymin=40 ymax=43 n=10 s=below TW=1,4,8,12
    """

    # Define the study area boundaries
    xmin = None
    xmax = None
    ymin = None
    ymax = None

    # DEFAULT DEFINITIONS OF n, select and TW in case they're not defined in the function call
    # Defining n= number of selected resolutions and select= "below" or "above" the second breakpoint, for default n= None and select= None
    n = 5
    select = None

    # Define time intervals for temporal resolutions
    # TW = (1, 4, 8, 12)
    TW = [1, 5, 10]  # for default

    # Check for argument of form n=VALUE
    for arg in sys.argv[2:]:
        print(str(arg))
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key == "n":
                n = int(value)
            elif key == "xmin":
                xmin = int(value)
            elif key == "xmax":
                xmax = int(value)
            elif key == "ymin":
                ymin = int(value)
            elif key == "ymax":
                ymax = int(value)
            elif key == "TW":
                # Remove parentheses and spaces, then split by comma
                value = value.strip("() ")
                try:
                    TW = [int(x.strip()) for x in value.split(",") if x.strip()]
                except ValueError:
                    print("Error: TW must be a list of integers in the form 'TW=x,y,z,...'")
                    sys.exit(1)
            elif key == "s":
                value = value.strip().lower()
                if value in ("below", "above", "none"):
                    select = None if value == "none" else value
                else:
                    print("Error: s must be 'below', 'above', or 'none'.")
                    sys.exit(1)

    if xmin is None or xmax is None or ymin is None or ymax is None:
        print(
            f"xmin [{xmin}], xmax [{xmax}], ymin [{ymin}], ymax [{ymax}] MUST be defined"
        )
        sys.exit(1)

    # Read the filtered data
    df = pd.read_csv(sys.argv[1])
    print(df.columns.tolist())
    base_name = Path(sys.argv[1]).stem
    safe_base = re.sub(r"[^A-Za-z0-9._-]", "_", base_name)

    # -------------------------------
    # START THE PROCEESSING DATA
    # -------------------------------
    filter_d = df[
        (df["lon"] >= xmin)
        & (df["lon"] <= xmax)
        & (df["lat"] >= ymin)
        & (df["lat"] <= ymax)
    ].copy()
    print(filter_d.columns.tolist())

    # Save the filtered data to a new CSV file
    output_file = f"filtered_{safe_base}.csv"
    filter_d.to_csv(output_file, index=False)

    # ---------------------------------------------------------------
    # Iterate over all grid sizes and calculate diversity indices
    # ---------------------------------------------------------------
    number_lon = xmax - xmin
    lon_sizes = divisors(number_lon)
    number_lat = ymax - ymin
    lat_sizes = divisors(number_lat)
    # Initialize the size_counts dictionary to keep track of the number of grids created for each size
    # Initialize the final results list to store the results for each grid size
    size_counts = {}
    final_results = []

    # Debug prints for light progress monitoring during grid generation
    total_combinations = len(lat_sizes) * len(lon_sizes)
    print(f"[DEBUG] Total grid combinations: {total_combinations}")
    counter = 0

    # Iterate over all combinations of lat and lon sizes
    for lat_size in lat_sizes:
        for lon_size in lon_sizes:
            counter += 1
            if counter == 1 or counter % 10 == 0:
                print(f"[DEBUG] Grid {counter}/{total_combinations} -> lat={lat_size}, lon={lon_size}")

            grid_df, size_counts = generate_all_grids(
                xmin, ymin, xmax, ymax, lon_size, lat_size, size_counts)  # Generate the grid for the current lat and lon sizes
            if grid_df.empty:
                continue

            assigned_points = {}

            filter_d["grid_cell"] = filter_d.apply(lambda row: assign_to_grid(row, grid_df, assigned_points), axis=1)  # Assign grid cells to points
            df_filtered = filter_d.dropna(subset=["grid_cell"]).copy()  # Remove points that were not assigned to any grid cell
            diversity_df = diversity_indices(df_filtered)# Calculate diversity indices for the grid cells
            grid_meta= grid_df.reset_index().rename(columns={'index': 'grid_cell'})
            merged = grid_meta.merge(diversity_df, on='grid_cell', how='left')  # Merge grid metadata with diversity indices
            merged["lat_size"] = lat_size
            merged["lon_size"] = lon_size
            final_results.append(merged)  # Append the results to the final results list

    # Concatenate all results into a single DataFrame
    if final_results:
        final_df = pd.concat(final_results, ignore_index=True)  # Reset the index 
        output_file_df = f"resolutions_&_indices_{safe_base}.csv" # Save the final results to a CSV file
        final_df.to_csv(output_file_df, index=False)
        print(f"Saved results {output_file_df}")
    else:
        print("No results to process.")
        sys.exit(1) # Exit if no results
   
    # Test the area limits
    print(final_df["area_km2"].min())
    print(final_df["area_km2"].max())
    
    # -------------------------------------------------
    # Calculate statistics for the final DataFrame
    # -------------------------------------------------
    stats_df = statistics(final_df)
    print(stats_df)
    # Save the statistics DataFrame to a CSV file
    output_file_stats = f"statistics_{safe_base}.csv"
    stats_df.to_csv(output_file_stats, index=False)
    print(f"Saved results {output_file_stats}")

    # --------------------------------------------------------------------------
    # Segmented regression Analysis to find the spatial resolutions candidates
    # --------------------------------------------------------------------------
    # Step 1: Define the limits for the segmentation, following the species-area realtionship curve
    # Filter the DataFrame based on area limits for segmentation
    xmin_a = 1
    xmax_a = stats_df["means_area_km2"].max()
    stats_df = stats_df[
        (stats_df["means_area_km2"] >= xmin_a) & (stats_df["means_area_km2"] <= xmax_a)
    ].copy()
    print(xmin_a, xmax_a)

    x = stats_df["means_area_km2"]
    x = np.array(x)  # Convert 'means_area_km2' to a numpy array
    y = stats_df["mean_r"]
    y = np.array(y)  # Convert 'mean_r' to a numpy array
    sizes1 = stats_df["size"]  # Convert 'size' to a string for labels

    # Fit the piecewise linear regression model using pwlf
    pwlf_model = pwlf.PiecewiseLinFit(x, y)  # Create the model

    # Estimate the breakpoints using the pwlf library, first segment
    breaks = pwlf_model.fit(2)  # 2 segments, 1 breakpoint
    breaks = np.clip(
        breaks, xmin_a, xmax_a
    )  # ensure that breakpoints are into the range
    breaks = [round(float(b), None) for b in breaks]
    print(breaks)
    x_fit = np.linspace(
        xmin_a, xmax_a, 1500
    )  # Create a range of x values for the fitted model
    y_fit = pwlf_model.predict(x_fit)  # Predict the y values for the fitted model

    # Step 2: Repeat the segmented regression for the first segment only
    xmax_a1 = breaks[1]
    print(xmax_a1)
    stats_df_b1 = stats_df[
        (stats_df["means_area_km2"] >= xmin_a) & (stats_df["means_area_km2"] <= xmax_a1)
    ]
    print(stats_df_b1)
    x1 = stats_df_b1["means_area_km2"]
    x1 = np.array(x1)  # Convert 'means_area_km2' to a numpy array
    y1 = stats_df_b1["mean_r"]
    y1 = np.array(y1)  # Convert 'mean_r' to a numpy array
    sizes2 = stats_df_b1["size"]  # Convert 'size' to a string for labels
    pwlf_model1 = pwlf.PiecewiseLinFit(x1, y1)
    breaks1 = pwlf_model1.fit(2)  # 2 segments, 1 breakpoint
    breaks1 = np.clip(
        breaks1, xmin_a, xmax_a1
    )  # ensure that breakpoints are into the range
    breaks1 = [
        round(float(b), None) for b in breaks1
    ]  # keep as float, rounded to 2 decimals
    print(breaks1)
    x_fit1 = np.linspace(
        xmin_a, xmax_a1, 1500
    )  # Create a range of x values for the fitted model
    y_fit1 = pwlf_model1.predict(x_fit1)  # Predict the y values for the fitted model

    # Plot segmented regression for both steps (step 1 and step 2)
    segmented_data = [
        {
            "x": x,
            "y": y,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "breaks": breaks,
            "sizes": sizes1,
            "title": "Piecewise Linear Fit (Step 1)",
        },
        {
            "x": x1,
            "y": y1,
            "x_fit": x_fit1,
            "y_fit": y_fit1,
            "breaks": breaks1,
            "sizes": sizes2,
            "title": "Piecewise Linear Fit (Step 2)",
        },
    ]

    for seg in segmented_data:
        plt.figure(figsize=(8, 6))
        # Assign labels "size" to the points
        for i in range(len(seg["x"])):
            plt.annotate(
                str(seg["sizes"].iloc[i]),
                (seg["x"][i], seg["y"][i]),
                textcoords="offset points",
                xytext=(5, 5),
                ha="right",
                fontsize=8,
                color="darkred",
            )
        plt.plot(
            seg["x_fit"], seg["y_fit"], color="green", label="Piecewise Linear Fit"
        )
        # Plot the breakpoints
        for br in seg["breaks"][1:-1]:
            plt.axvline(x=br, color="red", linestyle="--", label=f"Breakpoint: {br}")
        plt.xlabel("Area in km²")
        plt.ylabel("Nº of Species")
        x_ticks = np.linspace(1, max(seg["x"]), num=10, dtype=int)
        if 1 not in x_ticks:
            x_ticks = np.insert(x_ticks, 0, 1)
        plt.xticks(x_ticks)
        plt.legend()
        plt.savefig(f"Breakpoint_{br}.png", dpi=FIG_DPI)
        plt.close()

    # Print and save the second breakpoint results
    second_breakpoint = breaks1[1]
    print(f"The second breakpoint is at area: {second_breakpoint} km2")

    closest_resolutions = selected_resolutions(stats_df, second_breakpoint, n, select)
    print(closest_resolutions)

    # Distance in absolute value of ymax-ymin and xmax-xmin
    distance_lat = abs(ymax - ymin)
    distance_lon = abs(xmax - xmin)
    half_distance_lat = distance_lat / 2
    half_distance_lon = distance_lon / 2
    print(half_distance_lat)
    print(half_distance_lon)

    # Discar closet_resolutions with lat_size = distance_lat or lon_size = distance_lon
    final_closest_resolutions = closest_resolutions[
        (closest_resolutions["lat_size"] != distance_lat)
        & (closest_resolutions["lon_size"] != distance_lon)
        & (closest_resolutions["lat_size"] != half_distance_lat)
        & (closest_resolutions["lon_size"] != half_distance_lon)
    ]
    print(final_closest_resolutions)

    resolutions = final_closest_resolutions["size"].tolist()
    print(resolutions)
    # -----------------------------------------------------------------------------------------
    # Filter selection resolutions and calculate the number of polygons with richness = 0
    # -----------------------------------------------------------------------------------------
    resolutions_df = final_df[final_df["size"].isin(resolutions)].copy()
    print(resolutions_df)

    # Count the number of records where 'richness' == 0 or NaN for each 'size' in filtered_df
    zero_species_counts = (
        resolutions_df[resolutions_df["richness"] == 0]
        .groupby("size")
        .size()
        .reset_index(name="polygons with 0 species")
    )
    spatial_resolution_df = final_closest_resolutions.merge(
        zero_species_counts, on="size", how="left"
    )  # Merge the counts with the DataFrame
    spatial_resolution_df["polygons with 0 species"] = (
        spatial_resolution_df["polygons with 0 species"].fillna(0).astype(int)
    )  # Replace NaN with 0 and convert to int
    nan_species_counts = (
        resolutions_df[resolutions_df["richness"].isna()]
        .groupby("size")
        .size()
        .reset_index(name="polygons_with_NaN")
    )
    spatial_resolution_df = spatial_resolution_df.merge(
        nan_species_counts, on="size", how="left"
    )  # Merge the counts with the DataFrame
    spatial_resolution_df["polygons_with_NaN"] = (
        spatial_resolution_df["polygons_with_NaN"].fillna(0).astype(int)
    )  # Replace NaN with 0 and convert to int
    print(spatial_resolution_df)

    # Save the filtered DataFrame to a CSV file
    output_select_df = f"SR_Summary_table_{safe_base}.csv"
    spatial_resolution_df.to_csv(output_select_df, index=False)
    print(f"Saved results {output_select_df}")
    print("END of part 1: Selecting Spatial Resolutions")

    # Build the map for each size
    # Define size labels for plotting
    size_labels = {
        row["size"]: f"{row['lat_size']}x{row['lon_size']}"
        for _, row in spatial_resolution_df.iterrows()
    }
    filtered_df = final_df[final_df["size"].isin(resolutions)].copy()
    print(filtered_df)
    print(filtered_df["size"].unique())
    # Set fixed vmin and vmax for all maps to ensure the same color scale
    vmin = filtered_df["richness"].min()
    vmax = filtered_df["richness"].max()
    print(vmin, vmax)

    for size in sizes1:
        df_size = filtered_df[filtered_df["size"] == size]

        if df_size.empty:
            continue

        norm = mpl.colors.Normalize(
            vmin=vmin, vmax=vmax
        )  # Fixed normalization for all figures
        cmap = plt.cm.viridis_r  # reverse viridis colormap
        fig, ax = plt.subplots(figsize=(10, 8))

        for _, row in df_size.iterrows():
            x_coords = [
                row["lon_min"],
                row["lon_min"],
                row["lon_max"],
                row["lon_max"],
                row["lon_min"],
            ]
            y_coords = [
                row["lat_min"],
                row["lat_max"],
                row["lat_max"],
                row["lat_min"],
                row["lat_min"],
            ]
            color = cmap(norm(row["richness"]))
            ax.fill(
                x_coords,
                y_coords,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.2,
            )

        # legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Species richness")

        ax.set_title(
            f"Spatial distribution of species richness\nResolution {size}, grid = {size_labels[size]}",
            fontsize=12,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"Resolution_{size}.png", dpi=FIG_DPI)
        plt.close()
    # END OF PART 1 "SELETING SPATIAL RESOLUTIONS"

    # START OF PART 2: SPATIAL AND TEMPORAL ANALYSIS
    # --------------------------------------------------------------------------
    # TEMPORAL ANALYSIS for selected sizes. Generate the grid of selected resolutions + years. TEMPORAL ANALYSIS
    # -------------------------------------------------------------------------
    # Read the original dataframe "filter_d" with all points in the study area
    # df= pd.read_csv(f"filtered_{safe_base}.csv")
    # print(df.head())
    # print(df.columns.tolist())
    #Read SR_Summary_table to get the selected resolutions and their lat_size and lon_size to generate the grids for the temporal analysis
    # spatial_resolution_df = pd.read_csv(f"SR_Summary_table_{safe_base}.csv")
    # print(spatial_resolution_df.head())

    df = filter_d.copy()
    print(df.head())

    richness_year = (
        df.groupby(["year", "lat", "lon"])["species"]
        .nunique()
        .reset_index(name="richness")
    )
    cv_richness_y = (
        richness_year.groupby(["year"])["richness"]
        .apply(calculate_cv)
        .reset_index(name="cv_richness")
    )
    cv_richness_mean = cv_richness_y["cv_richness"].mean()
    cv_richness_std = cv_richness_y["cv_richness"].std()
    ucl = cv_richness_mean + cv_richness_std
    lcl = cv_richness_mean - cv_richness_std
    control_limits_df = pd.DataFrame([{"cv_richness_mean": cv_richness_mean, "cv_richness_std": cv_richness_std, "ucl": ucl, "lcl": lcl}])
    control_limits_df.to_csv(f"Control_Limits_{safe_base}.csv",index=False)
    print(
        f"CV Mean: {cv_richness_mean}, CV Std: {cv_richness_std}, UCL: {ucl}, LCL: {lcl}"
    )

    # -------------------------------------------------------------------------------------------------------------------
    # Generate FOR EACH resolution with TEMPORAL WINDOWS for default: 1, 5, and 10 years, and calculate diversity indices
    # -------------------------------------------------------------------------------------------------------------------
    size_labels_map = {
        (row["lat_size"], row["lon_size"]): row["size"]
        for _, row in spatial_resolution_df.iterrows()
    }
    print(size_labels_map)

    print(TW)

    combinations = [
        (row["lat_size"], row["lon_size"])
        for _, row in spatial_resolution_df.iterrows()
    ]
    min_year, max_year = df["year"].min(), df["year"].max()
    all_grids_results = []

    for lat_size, lon_size in combinations:
        print(f"Processing SR: lat={lat_size}, lon={lon_size}")
        for interval in TW:
            print(f"  Processing TW: {interval} years")
            year_groups = list(range(min_year, max_year + 1, interval))
            for start_year in year_groups:
                end_year = min(start_year + interval - 1, max_year)
                print(f"    Period: {start_year}-{end_year}")
                df_interval = df[
                    (df["year"] >= start_year) & (df["year"] <= end_year)
                ].copy()
                if df_interval.empty:
                    continue
                df_interval["period"] = f"{start_year}_{end_year}"
                grid_df_ys, _ = generate_period_grids(xmin, ymin, xmax, ymax,
                    [lat_size],
                    [lon_size],
                    list(range(start_year, end_year + 1)),
                    size_labels_map=size_labels_map,
                )
                assigned_points = {}
                df_interval["grid_cell"] = df_interval.apply(
                    lambda row: assign_to_grid_p(row, grid_df_ys, assigned_points),
                    axis=1,
                )
                df_result = calculate_diversity_indices_p(df_interval)
                df_result["period_grid_cell"] = (
                    df_result["period"] + "_" + df_result["grid_cell"].astype(str)
                )
                grid_df_ys["period"] = f"{start_year}_{end_year}"
                grid_df_ys["period_grid_cell"] = (
                    grid_df_ys["period"] + "_" + grid_df_ys["grid_cell"].astype(str)
                )
                for index in [
                    "richness",
                    "abundance",
                    "shannon",
                    "simpson",
                    "margalef",
                    "menhinick",
                ]:
                    grid_df_ys[index] = (
                        grid_df_ys["period_grid_cell"]
                        .map(df_result.set_index("period_grid_cell")[index])
                        .fillna(0)
                    )
                grid_df_ys["lat_size"] = lat_size
                grid_df_ys["lon_size"] = lon_size
                grid_df_ys["time_resolution"] = interval
                all_grids_results.append(grid_df_ys)

    final_df_time_All = pd.concat(all_grids_results, ignore_index=True)
    print(final_df_time_All.columns.tolist())

    output_file_time = f"time_series_{safe_base}.csv"
    final_df_time_All.to_csv(output_file_time, index=False)
    print(f"Saved results {output_file_time}")

    # CONTROL CHART: Plot the Coefficient of Variation (CV) of Species Richness over time for each selected size and time resolution
    select_df_no_zeros = final_df_time_All[
        final_df_time_All["richness"] > 0
    ].copy()  # Filter out rows with richness = 0 necessary for CV calculation

    sizes = final_df_time_All["size"].unique().tolist()

    colors = color_for_period(TW)
    select_df_no_zeros["middle_year"] = select_df_no_zeros["period"].apply(
        lambda x: int((int(str(x).split("_")[0]) + int(str(x).split("_")[-1])) / 2)
    )
    xticks_years = list(range(min_year, max_year, 5))
    if max_year not in xticks_years:
        xticks_years.append(max_year)
    
    # y-axis limits for CV plot
    all_cv_values = []
    for size in sizes:
        for period in TW:
            df_size_period = select_df_no_zeros[
                (select_df_no_zeros["size"] == size)
                & (select_df_no_zeros["time_resolution"] == period)
            ].copy()
            if df_size_period.empty:
                continue
            x_col = "year" if period == 1 else "middle_year"
            cv_tmp =(df_size_period.groupby(x_col)["richness"].apply(calculate_cv).reset_index(name="cv_richness"))
            vals = cv_tmp["cv_richness"].replace([np.inf, -np.inf], np.nan).dropna().values
            if len(vals) > 0:
                all_cv_values.extend(vals.tolist())
    if len(all_cv_values) == 0:
        raise ValueError("No CV values calculated. Check the data and filtering criteria.")
    
    # Base min/max from plotted CVs
    cvymin = min(all_cv_values)
    cvymax = max(all_cv_values)
    
    # Ensure control lines always fit too
    cvymin=min(cvymin, lcl, cv_richness_mean)
    cvymax=max(cvymax, ucl, cv_richness_mean)
    
    # Add margin so points/lines are not clipped
    yrange = cvymax - cvymin
    pad = 0.08 *yrange if yrange > 0 else  0.1 # 8% padding or 0.1 if all values are the same
    Y_MIN_GLOBAL = cvymin - pad
    Y_MAX_GLOBAL = cvymax + pad

    for size in sizes:
        plt.figure(figsize=(12, 6))
        has_data = False
        for period in TW:
            df_size_period = select_df_no_zeros[
                (select_df_no_zeros["size"] == size)
                & (select_df_no_zeros["time_resolution"] == period)
            ].copy()
            if df_size_period.empty:
                print(f"Without data for {size}.")
                continue
            x_col = "year" if period == 1 else "middle_year"
            cv_df = (
                df_size_period.groupby(x_col)["richness"]
                .apply(calculate_cv)
                .reset_index(name="cv_richness")
            )
            plt.plot(
                cv_df[x_col],
                cv_df["cv_richness"],
                marker="o",
                label=f"Periodo {period}",
                color=colors[period],
            )
        
            plt.axhline(
                cv_richness_mean,
                color="black",
                linestyle="--",
                label=f"Mean CV: {cv_richness_mean:.4f}",
            )
            plt.axhline(ucl, color="red", linestyle=":", label="UCL & LCL")
            plt.axhline(lcl, color="red", linestyle=":")
            has_data = True
        if has_data:
            plt.title(f"Coefficient of Variation of Species Richness {size}")
            plt.xlabel("Year")
            plt.ylabel("Coefficient of Variation (CV)")
            plt.ylim(Y_MIN_GLOBAL, Y_MAX_GLOBAL)
            plt.xticks(xticks_years, rotation=90)
            plt.grid(False)
            plt.tight_layout()
            plt.savefig(f"CV_Richness_{size}.png", dpi=FIG_DPI)
            plt.close()

    summary_list = []
    for size in sizes:
        for period in TW:
            df_filtered = select_df_no_zeros[
                (select_df_no_zeros["size"] == size)
                & (select_df_no_zeros["time_resolution"] == period)
            ].copy()
            if df_filtered.empty:
                outliers_count = 0
                mean_cv = None
                diff_cv = None
            else:
                x_col = "year" if period == 1 else "middle_year"
                cv_df = (
                    df_filtered.groupby(x_col)["richness"]
                    .apply(calculate_cv)
                    .reset_index(name="cv_richness")
                )
                mean_cv = cv_df["cv_richness"].mean()
                diff_cv = mean_cv - cv_richness_mean
                outliers_count = len(cv_df[
                    (cv_df["cv_richness"] > ucl) | (cv_df["cv_richness"] < lcl)
                ])
            summary_list.append(
                {
                    "size": size,
                    "time_resolution": period,
                    "mean_cv_richness": mean_cv,
                    "diff_cv_richness": diff_cv,
                    "outliers_count": outliers_count,
                }
            )

    summary_df_CV_time = pd.DataFrame(summary_list)
    print(summary_df_CV_time)

    # Obtain the average of outliers_count for each size across time resolutions
    avg_outliers = (
        summary_df_CV_time.groupby("size")["outliers_count"]
        .mean()
        .reset_index(name="avg_outliers_count")
    )
    # select the size with the lowest average outliers_count
    best_size_row = avg_outliers.loc[avg_outliers["avg_outliers_count"].idxmin()]
    best_size = best_size_row["size"]
    # using the best_size, filter the summary_df_CV_time to get the corresponding rows
    best_size_summary = summary_df_CV_time[summary_df_CV_time["size"] == best_size]

    # Apply the selection criteria
    optimal_combinations_df = select_optimal_combinations(best_size_summary)
    print(optimal_combinations_df)
    
    # Partial result for all spatial-temporal combinations
    output_file_summary_CV = f"Summary_time_CV_{safe_base}.csv"
    summary_df_CV_time.to_csv(output_file_summary_CV, index=False)
    print(f"Saved results {output_file_summary_CV}")
    
    # Save the dataframe of final optimal combinations to a CSV file
    sr= optimal_combinations_df["size"]
    time_w= optimal_combinations_df["time_resolution"]
    df_OpR_partial = final_df_time_All[
        (final_df_time_All["size"].isin(sr))
        & (final_df_time_All["time_resolution"].isin(time_w))
    ].copy()
    id_cols = ["lat_size","lon_size","lon_min","lat_min","lon_max","lat_max","area_km2","size","grid_cell","period","period_grid_cell","time_resolution"]
    df_OpR = (df_OpR_partial.sort_values("year").drop_duplicates(subset=id_cols, keep="first").reset_index(drop=True))
    metric_cols = ["richness", "abundance", "shannon", "simpson", "margalef", "menhinick"]
    df_OpR[metric_cols] = df_OpR[metric_cols].replace(0, np.nan)
    print(df_OpR.columns.tolist())
    output_file_OpR_df = f"OpR_dataset_{safe_base}.csv"
    df_OpR.to_csv(output_file_OpR_df, index=False)
    print(f"Saved results {output_file_OpR_df}")

    # Final optimal (OpR) combinations saved to CSV
    output_file_OpR = f"OpR_{safe_base}.csv"
    optimal_combinations_df.to_csv(output_file_OpR, index=False)
    print(f"Saved results {output_file_OpR}")

if __name__ == "__main__":
    start = perf_counter()
    main()
    stop = perf_counter()
    print("time taken:", stop - start)
