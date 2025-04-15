import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import glob


def compute_average_from_file(filename):
    """
    Reads the given file line by line, parses each line as a float,
    and returns the arithmetic mean of all values.
    """
    values = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:  # skip empty lines
                values.append(float(line))

    return sum(values) / len(values)


def compute_sum_for_c0_d_ge_1(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path, index_col=0)

    # Ensure index (d_values) and columns (c_values) are numeric
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)

    # Select values for c=0 and d>=1
    c_value = 0  # Fixed c value

    filtered_values = df.loc[
        df.index >= 1, c_value
    ]  # Filter rows where d >= 1 and column c=0

    # Compute the average of all d values for c=0
    weights = df[c_value].values
    average_d = np.average(df.index, weights=weights)

    # Compute and return the sum
    return filtered_values.sum(), average_d


def process_file(file_path, statistic="average"):
    # Load the CSV into a DataFrame
    df = pd.read_csv(file_path, index_col=0)

    # Get the d values (row indices)
    d_values = df.index.astype(float)
    c_values = df.columns.astype(float)

    # Compute the statistic (average or median) of d for each c
    results_c = {}
    for c in df.columns:
        weights = df[c].values
        if statistic == "average":
            stat_d = np.average(d_values, weights=weights)
        elif statistic == "median":
            # Create an expanded array of d values based on weights
            expanded = np.repeat(d_values, weights.astype(int))
            stat_d = np.median(expanded) if len(expanded) > 0 else np.nan
        results_c[float(c)] = stat_d

    # DEPRECATED

    """
    results_d = {}
    for d in df.index:
        weights = df.loc[d].values
        if statistic == 'average':
            stat_c = np.average(c_values, weights=weights)
        elif statistic == 'median':
            # Create an expanded array of d values based on weights
            expanded = np.repeat(c_values, weights.astype(int))
            stat_c = np.median(expanded) if len(expanded) > 0 else np.nan
        results_d[float(d)] = stat_c
    """

    # CONTINUE HERE
    results_d = -1
    for d in df.index:
        if float(d) >= 1:  # Only consider rows where d >= 1
            row = df.loc[d].values
            # Find the smallest c where all corresponding d >= 1 entries are zero
            non_zero_c = [c_values[i] for i, value in enumerate(row) if value > 0]
            if non_zero_c:
                results_d = max(results_d, max(non_zero_c))

    # Return the results as a sorted list of tuples (c, statistic_d)
    return sorted(results_c.items()), results_d


def extract_parameters(file_name):
    # Extract mu, ga, and de values from the filename
    match = re.search(
        r"disc_(\d+)_mu_(\d+\.\d+)_ga_(\d+\.\d+)_de_(\d+\.\d+)", file_name
    )
    if match:
        disc, mu, ga, de = match.groups()
        return float(disc), float(mu), float(ga), float(de)
    return None, None, None, None


def process_satisfaction_file(file_path_satisfaction):
    """Compute the average of row minima and row averages from a satisfaction file."""
    df = pd.read_csv(file_path_satisfaction, header=None)  # Load satisfaction data
    avg_row_minima = df.min(axis=1).mean()
    avg_row_averages = df.mean(axis=1).mean()
    std_row_averages = df.mean(axis=1).std()

    # Replace df with your actual DataFrame
    # Compute the bottom 10th percentile mean for each row
    bottom_10th_means = df.apply(
        lambda row: row.nsmallest(max(1, int(len(row) * 0.1))).mean(), axis=1
    )

    # Compute the average of these means across rows
    avg_bottom_10th_percentile = bottom_10th_means.mean()
    std_bottom_10th_percentile = bottom_10th_means.std()

    return (
        avg_bottom_10th_percentile,
        std_bottom_10th_percentile,
        avg_row_averages,
        std_row_averages,
    )


def extract_variant(name):
    if "MW" in name:
        return "MW"
        # Extract mu, ga, and de values from the filename
    match = re.search(r"version_(\d)_", name)
    if match:
        vn = match.group(1)  # Use group(1) to access the first capturing group
        if vn == "0":
            return "simple"
        elif vn == "2":
            return "complex"
    print("ERROR")
    return None


def generate_latex_plot(config, c_values, d_values_dict, min_c, min_d, x_max, y_max):
    """
    Generates a standalone LaTeX file with a plot using pgfplots.

    Parameters:
        config (str): The configuration name.
        c_values (list): The x-axis values (c values).
        d_values_dict (dict): A dictionary with variants as keys and their y-axis values (d values).
        min_c (float): Minimum value of c for shading.
        min_d (float): Minimum value of d for shading.
        x_max (float): Maximum value for the x-axis.
        y_max (float): Maximum value for the y-axis.
    """
    latex_file = f"{config}.tex"

    # Start the LaTeX file content
    latex_content = r"""
\begin{tikzpicture}
\begin{axis}[
    width=10cm,
    height=6cm,
    xlabel={$c$},
    ylabel={Max $d$ with $(c,d)$-violation},
    grid=both,
    legend style={at={(0.5,-0.2)},anchor=north,legend columns=-1},
    ymin=-0.1,
    xmin=-0.1,
"""

    latex_content += f""" ymax={y_max}, xmax={x_max}]"""

    # Add gray rectangle for the shaded area
    latex_content += f"""
\\addplot[draw=none, fill=gray, fill opacity=0.3] coordinates {{
    ({min_c}, {min_d}) ({min_c}, {y_max}) ({x_max}, {y_max}) ({x_max}, {min_d}) ({min_c}, {min_d})
}};
"""

    # Add each variant's data to the plot
    for variant in sorted(d_values_dict.keys()):
        d_values = d_values_dict[variant]
        color = {"simple": "blue", "complex": "green", "MW": "orange"}.get(
            variant, "black"
        )
        latex_content += f"""
    \\addplot[
        color={color},
        legend entry={variant},
        mark=*,
        thick
    ] coordinates {{
    """
        for c, d in zip(c_values, d_values):
            latex_content += f"({c}, {d}) "
        latex_content += """};\n"""

    # End the LaTeX file content
    latex_content += r"""
    \end{axis}
    \end{tikzpicture}
    """

    # Write the LaTeX content to a file
    with open(latex_file, "w") as f:
        f.write(latex_content)
    print(f"LaTeX plot saved as '{latex_file}'")


# folder="approx_dev_5_reso"
folder = "approx_gen_error_worst"
# folder="d_invest"
# folder="approx_final"

file_paths = glob.glob("./" + folder + "/heatmap*.csv")

# file_names=[os.path.basename(p) for p in file_paths]


from collections import defaultdict


# Function to extract configuration from file name
def get_config(file_name):
    config = re.match(r"(.*_version)", file_name).group(1)
    return config


# Group files by configuration
grouped_files = defaultdict(list)

for path in file_paths:
    config = get_config(path)  # Get the configuration
    grouped_files[config].append(path)  # Group files by configuration

# Output grouped files

for config, files in grouped_files.items():
    print(f"Configuration: {config}")
    for file in files:
        print(f"  - {file}")

    # Initialize a plot
    plt.figure(figsize=(10, 6))

    d_values_dict = {}

    # Process each CSV file and plot the average `d` values
    for file_path in files:
        # Extract the file name for labeling
        file_name = os.path.basename(file_path)

        disc, mu, ga, de = extract_parameters(file_name)
        min_c = 2 * disc + de
        min_d = 1 / (ga * mu)
        variant = extract_variant(file_name)

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col=0)

        # Extract `d_values` and `c_values`
        d_values = df.index.astype(float)
        c_values = df.columns.astype(float)

        ds = []
        cs = []
        for c in df.columns:
            weights = df[c].values
            stat_d = np.average(d_values, weights=weights)
            ds.append(stat_d)
            cs.append(float(c))

        d_values_dict[variant] = ds

        # Compute average `d` values for each `c` value
        print(cs, ds)
        # Plot the average `d` values
        if variant == "simple":
            color = "blue"
        elif variant == "complex":
            color = "green"
        elif variant == "MW":
            color = "orange"
        plt.plot(
            cs,
            ds,
            label=variant,  # Use file name for the legend
            marker="o",
            linestyle="-",
            color=color,
        )

        # Define the plot limits
    x_min = min(cs)
    x_max = max(cs)
    y_min = min(min(d_values) for d_values in d_values_dict.values())
    y_max = max(max(d_values) for d_values in d_values_dict.values())

    y_max = max(y_max * 1.1, min_d * 1.1)
    x_max = x_max + 0.1

    # Add shaded region for x > min_c and y > min_d
    plt.axvspan(
        min_c,
        x_max,
        ymin=(min_d - y_min) / (y_max - y_min),
        color="lightgray",
        alpha=0.5,
        label="x > min_c and y > min_d",
    )

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Customize the plot
    config_key = (
        f"be={disc} mu={mu} ga={ga} de={de} "
        + variant
        + " "
        + str(min_c)
        + " "
        + str(min_d)
    )
    plt.title(config_key)
    plt.xlabel("c")
    plt.ylabel("Average d values")
    plt.grid(True)
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()

    # Save the plot or display it
    plt.savefig(config + ".png")
    plt.show()
    generate_latex_plot(config, cs, d_values_dict, min_c, min_d, x_max, y_max)

import csv
from collections import defaultdict

# Initialize a dictionary to group data by configuration
grouped_data = defaultdict(lambda: {"MW": {}, "simple": {}, "complex": {}})

# Process the CSV file rows
for path in file_paths:
    file_name = os.path.basename(path)
    variant = extract_variant(file_name)
    disc, mu, ga, de = extract_parameters(file_name)
    config_key = (
        f"be={disc} mu={mu} ga={ga} de={de}"  # Grouping key based on configuration
    )
    path_rep = path.replace("heatmap_error", "satislist")
    path_minc = path.replace("heatmap_error_disc", "minc_nodev")
    # print(path_rep)
    file_path_satisfaction = os.path.basename(path_rep)
    # print(file_path_satisfaction)
    # Compute metrics
    avg_row_minima, std_minima, avg_row_averages, std_average = (
        process_satisfaction_file(path_rep)
    )
    BJR_violations, average_d_forc0 = compute_sum_for_c0_d_ge_1(path)
    minc_nodev = compute_average_from_file(path_minc)

    # Store metrics in the grouped dictionary
    grouped_data[config_key][variant] = {
        "Average Satisfaction": avg_row_averages,
        "Std Satisfaction": std_average,
        "Average Minimum Satisfaction": avg_row_minima,
        "Std Minimum Satisfaction": std_minima,
        "Number BJR violations": BJR_violations,
        "minc_nodev": minc_nodev,
        "mind_nodev": average_d_forc0,
    }

# Create the adjusted CSV structure with multi-level headers
csvfile = [
    [
        "Configuration",
        "Average Satisfaction",
        "Average Satisfaction",
        "Average Satisfaction",
        "Average Minimum Satisfaction",
        "Average Minimum Satisfaction",
        "Average Minimum Satisfaction",
        "BJR violations",
        "BJR violations",
        "BJR violations",
        "Min C",
        "Min D",
    ],
    [
        "",
        "MW",
        "Simple",
        "Complex",
        "MW",
        "Simple",
        "Complex",
        "MW",
        "Simple",
        "Complex",
        "Complex",
        "Complex",
    ],
]

for config, variants in grouped_data.items():
    row = [
        config,
        f"{variants['MW'].get('Average Satisfaction', '')} ± {variants['MW'].get('Std Satisfaction', '')}",
        f"{variants['simple'].get('Average Satisfaction', '')} ± {variants['simple'].get('Std Satisfaction', '')}",
        f"{variants['complex'].get('Average Satisfaction', '')} ± {variants['complex'].get('Std Satisfaction', '')}",
        f"{variants['MW'].get('Average Minimum Satisfaction', '')} ± {variants['MW'].get('Std Minimum Satisfaction', '')}",
        f"{variants['simple'].get('Average Minimum Satisfaction', '')} ± {variants['simple'].get('Std Minimum Satisfaction', '')}",
        f"{variants['complex'].get('Average Minimum Satisfaction', '')} ± {variants['complex'].get('Std Minimum Satisfaction', '')}",
        variants["MW"].get("Number BJR violations", ""),
        variants["simple"].get("Number BJR violations", ""),
        variants["complex"].get("Number BJR violations", ""),
        variants["complex"].get("minc_nodev", ""),
        variants["complex"].get("mind_nodev", ""),
    ]
    csvfile.append(row)

# Output file path
output_file = "./" + folder + "/adjusted_overview_with_multi_header.csv"

# Write the data to a CSV file row by row
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csvfile)

print(f"Adjusted CSV file with multi-header saved as '{output_file}'")

# Generate the LaTeX table with "Variant" above parameters
latex_table = "\\begin{table}[ht]\n\\centering\n"
latex_table += "\\begin{tabular}{cccc|ccc|ccc|ccc}\n"
latex_table += "\\hline\n"
latex_table += "\\multicolumn{4}{c|}{Variant} & \\multicolumn{3}{c|}{Average Satisfaction} & \\multicolumn{3}{c|}{Average Minimum Satisfaction} & \\multicolumn{3}{c}{BJR violations} \\\\\n"
latex_table += "$\\beta$ & $\\mu$ & $\\gamma$ & $\\delta$ & MW & Simple & Complex & MW & Simple & Complex & MW & Simple & Complex \\\\\n"
latex_table += "\\hline\n"

for config, variants in grouped_data.items():
    # Extract individual parameters from the configuration string
    beta = config.split(" ")[0].split("=")[1]
    mu = config.split(" ")[1].split("=")[1]
    gamma = config.split(" ")[2].split("=")[1]
    delta = config.split(" ")[3].split("=")[1]

    # Add rows for each configuration
    row = [
        beta,
        mu,
        gamma,
        delta,
        # f"{round(variants['MW'].get('Average Satisfaction', 0), 2)} ± {round(variants['MW'].get('Std Satisfaction', 0), 2)}" if "MW" in variants else "",
        # f"{round(variants['simple'].get('Average Satisfaction', 0), 2)} ± {round(variants['simple'].get('Std Satisfaction', 0), 2)}" if "simple" in variants else "",
        # f"{round(variants['complex'].get('Average Satisfaction', 0), 2)} ± {round(variants['complex'].get('Std Satisfaction', 0), 2)}" if "complex" in variants else "",
        # f"{round(variants['MW'].get('Average Minimum Satisfaction', 0), 2)} ± {round(variants['MW'].get('Std Minimum Satisfaction', 0), 2)}" if "MW" in variants else "",
        # f"{round(variants['simple'].get('Average Minimum Satisfaction', 0), 2)} ± {round(variants['simple'].get('Std Minimum Satisfaction', 0), 2)}" if "simple" in variants else "",
        # f"{round(variants['complex'].get('Average Minimum Satisfaction', 0), 2)} ± {round(variants['complex'].get('Std Minimum Satisfaction', 0), 2)}" if "complex" in variants else "",
        f"{round(variants['MW'].get('Average Satisfaction', 0), 2)}"
        if "MW" in variants
        else "",
        f"{round(variants['simple'].get('Average Satisfaction', 0), 2)}"
        if "simple" in variants
        else "",
        f"{round(variants['complex'].get('Average Satisfaction', 0), 2)}"
        if "complex" in variants
        else "",
        f"{round(variants['MW'].get('Average Minimum Satisfaction', 0), 2)}"
        if "MW" in variants
        else "",
        f"{round(variants['simple'].get('Average Minimum Satisfaction', 0), 2)}"
        if "simple" in variants
        else "",
        f"{round(variants['complex'].get('Average Minimum Satisfaction', 0), 2)}"
        if "complex" in variants
        else "",
        variants["MW"].get("Number BJR violations", "") if "MW" in variants else "",
        variants["simple"].get("Number BJR violations", "")
        if "simple" in variants
        else "",
        variants["complex"].get("Number BJR violations", "")
        if "complex" in variants
        else "",
    ]
    latex_table += " & ".join(map(str, row)) + " \\\\\n"

latex_table += "\\hline\n"
latex_table += "\\end{tabular}\n"
latex_table += "\\caption{Overview of metrics grouped by configuration and variant, with Variant label above parameters.}\n"
latex_table += "\\label{tab:metrics_overview_variant}\n"
latex_table += "\\end{table}\n"

# Save the LaTeX table to a file
output_file = "./" + folder + "/metrics_overview_with_variant_label.tex"
with open(output_file, "w") as f:
    f.write(latex_table)

print(f"LaTeX table with Variant label above parameters saved as '{output_file}'")


# BETA PLOT

# Filter entries where mu=1, ga=1, and de=0
filtered_data = {
    config: variants
    for config, variants in grouped_data.items()
    if "mu=1" in config and "ga=1" in config and "de=0" in config
}

# Extract beta (disc) values and corresponding minc_nodev metrics
beta_values = []
minc_values = []

for config, variants in filtered_data.items():
    beta = float(
        config.split(" ")[0].split("=")[1]
    )  # Extract beta (disc) from the config key
    minc_nodev = variants["complex"].get("minc_nodev")
    if minc_nodev != "":
        beta_values.append(beta)
        minc_values.append(minc_nodev)

# Sort the values by beta for a proper line plot
sorted_data = sorted(zip(beta_values, minc_values), key=lambda x: x[0])
beta_values, minc_values = zip(*sorted_data)

# Create the line plot
plt.figure(figsize=(8, 6))
plt.plot(beta_values, minc_values, marker="o", linestyle="-", label="minc_nodev")
plt.title("Growth of minc_nodev with Increasing Beta", fontsize=14)
plt.xlabel("Beta (disc)", fontsize=12)
plt.ylabel("minc_nodev", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("./" + folder + "/beta_change.jpg", bbox_inches="tight")


# Filter entries where be=0, mu=1, and ga=1
filtered_data = {
    config: variants
    for config, variants in grouped_data.items()
    if "be=0" in config and "mu=1" in config and "ga=1" in config
}

# Extract de values and corresponding minc_nodev metrics
de_values = []
minc_values = []

for config, variants in filtered_data.items():
    de = float(config.split(" ")[3].split("=")[1])  # Extract de from the config key
    minc_nodev = variants["complex"].get("minc_nodev")
    if minc_nodev != "":
        de_values.append(de)
        minc_values.append(minc_nodev)

# Sort the values by de for a proper line plot
sorted_data = sorted(zip(de_values, minc_values), key=lambda x: x[0])
de_values, minc_values = zip(*sorted_data)

# Create the line plot
plt.figure(figsize=(8, 6))
plt.plot(de_values, minc_values, marker="o", linestyle="-", label="minc_nodev")
plt.title("Impact of Changing de on minc_nodev", fontsize=14)
plt.xlabel("de", fontsize=12)
plt.ylabel("minc_nodev", fontsize=12)
plt.grid(True)
plt.legend(fontsize=12)
plt.savefig("./" + folder + "/delta_change.jpg", bbox_inches="tight")


# Filter entries where be=0, de=0, and ga=1
filtered_data_mu = {
    config: variants
    for config, variants in grouped_data.items()
    if "be=0" in config and "de=0" in config and "ga=1" in config
}

# Extract mu values and corresponding mind_nodev metrics
mu_values = []
mind_values_mu = []

for config, variants in filtered_data_mu.items():
    mu = float(config.split(" ")[1].split("=")[1])  # Extract mu from the config key
    mind_nodev = variants["complex"].get("mind_nodev")
    if mind_nodev != "":
        mu_values.append(mu)
        mind_values_mu.append(mind_nodev)

# Sort the values by mu for a proper line plot
sorted_data_mu = sorted(zip(mu_values, mind_values_mu), key=lambda x: x[0])
mu_values, mind_values_mu = zip(*sorted_data_mu)

# Filter entries where be=0, de=0, and mu=1
filtered_data_ga = {
    config: variants
    for config, variants in grouped_data.items()
    if "be=0" in config and "de=0" in config and "mu=1" in config
}

# Extract ga values and corresponding mind_nodev metrics
ga_values = []
mind_values_ga = []

for config, variants in filtered_data_ga.items():
    ga = float(config.split(" ")[2].split("=")[1])  # Extract ga from the config key
    mind_nodev = variants["complex"].get("mind_nodev")
    if mind_nodev != "":
        ga_values.append(ga)
        mind_values_ga.append(mind_nodev)

# Sort the values by ga for a proper line plot
sorted_data_ga = sorted(zip(ga_values, mind_values_ga), key=lambda x: x[0])
ga_values, mind_values_ga = zip(*sorted_data_ga)

# Generate values for 1/x curve
x_curve = np.linspace(
    min(min(mu_values), min(ga_values)), max(max(mu_values), max(ga_values)), 500
)
y_curve = 1 / x_curve

# Create a combined plot
plt.figure(figsize=(10, 8))

# Plot for mu
plt.plot(
    mu_values,
    mind_values_mu,
    marker="o",
    linestyle="-",
    label="mind_nodev (changing mu)",
)

# Plot for ga
plt.plot(
    ga_values,
    mind_values_ga,
    marker="s",
    linestyle="--",
    label="mind_nodev (changing ga)",
)

# Plot for 1/x curve
# plt.plot(x_curve, y_curve, linestyle="-.", color="gray", label="1/x curve")

# Add titles and labels
plt.title("Impact of Changing mu and ga on mind_nodev with 1/x Curve", fontsize=14)
plt.xlabel("Parameter Value (mu or ga)", fontsize=12)
plt.ylabel("mind_nodev", fontsize=12)
plt.gca().invert_xaxis()  # Flip the x-axis
plt.grid(True)
plt.legend(fontsize=12)

# Save the plot
plt.savefig(
    "./" + folder + "/combined_mu_ga_with_curve_flipped.jpg", bbox_inches="tight"
)
# plt.show()

"""
import csv


csvfile=[["version", "Average Satisfaction", "Average Minimum Satisfaction", "Number BJR violations"]]
for path in file_paths:
    file_name = os.path.basename(path)
    variant = extract_variant(file_name)
    disc, mu, ga, de = extract_parameters(file_name)
    name=f"be={disc} mu={mu} ga={ga} de={de} variant={variant}"
    file_path_satisfaction = file_name.replace("heatmap_error", "satislist")
    avg_row_minima, avg_row_averages = process_satisfaction_file(file_path_satisfaction)
    BJR_violations=compute_sum_for_c0_d_ge_1(path)
    csvfile.append([name,avg_row_averages,avg_row_minima,BJR_violations])





# Separate the header and the data
header = csvfile[0]
rows = csvfile[1:]

# Sort the rows by the first entry in each row
sorted_rows = sorted(rows, key=lambda row: row[0])  # Sort numerically

# Combine header and sorted rows
csvfile = [header] + sorted_rows



# Output file path
output_file = "overview.csv"

# Write the data to a CSV file row by row
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(csvfile)
"""


exit()


# List of CSV file paths
file_paths = glob.glob("./heatmap*.csv")
# Process files and sort by delta (de)
file_data = []
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    file_type = "_random" if "_random" in file_name else "_worst"
    file_path_satisfaction = file_name.replace("heatmap_error", "satislist")
    variant = extract_variant(file_name)
    disc, mu, ga, de = extract_parameters(file_name)
    averages_for_c, d_viol = process_file(file_path, statistic="average")
    # print("ERRRRR", d_viol)
    medians_for_c, _ = process_file(file_path, statistic="median")
    avg_row_minima, avg_row_averages = process_satisfaction_file(file_path_satisfaction)
    file_data.append(
        {
            "disc": disc,
            "mu": mu,
            "ga": ga,
            "de": de,
            "averages": averages_for_c,
            "medians": medians_for_c,
            "d_viol": d_viol,
            "avg_row_minima": avg_row_minima,
            "avg_row_averages": avg_row_averages,
            "file_name": file_name,
            "file_type": file_type,
            "variant": variant,
        }
    )

# Sort by delta (de)
file_data = sorted(file_data, key=lambda x: x["de"])

# Define linestyles for different de values
linestyles = ["solid", "dashed", "dotted", "dashdot"]
unique_de_values = sorted(set(item["de"] for item in file_data))
de_linestyle_map = {
    de: linestyles[i % len(linestyles)] for i, de in enumerate(unique_de_values)
}

# Plot 1: Average
plt.figure(figsize=(10, 6))
for data in file_data:
    c_values, avg_d_values = zip(*data["averages"])
    linestyle = de_linestyle_map[data["de"]]
    label = f"mu={data['mu']}, ga={data['ga']}, de={data['de']} ({data['file_type']})"
    plt.plot(c_values, avg_d_values, label=label, linestyle=linestyle)

plt.title("Average d Values vs. c for Different Files")
plt.xlabel("c")
plt.ylabel("Average d")
plt.legend(title="File Parameters")
plt.grid(True)
plt.tight_layout()
plt.savefig("average_d_vs_c.png")  # Save the plot as a PNG file
plt.close()

# Plot 2: Median
plt.figure(figsize=(10, 6))
for data in file_data:
    c_values, median_d_values = zip(*data["medians"])
    linestyle = de_linestyle_map[data["de"]]
    label = f"mu={data['mu']}, ga={data['ga']}, de={data['de']} ({data['file_type']})"
    plt.plot(c_values, median_d_values, label=label, linestyle=linestyle)

plt.title("Median d Values vs. c for Different Files")
plt.xlabel("c")
plt.ylabel("Median d")
plt.legend(title="File Parameters")
plt.grid(True)
plt.tight_layout()
plt.savefig("median_d_vs_c.png")  # Save the plot as a PNG file
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

# / Plot for mu and ga separately by file type
# /for file_type in ["_random", "_worst"]:
# /    if file_type == "_random":
# /        mark = 'o'
# /    else:
# /        mark = '^'
for file_type in ["MW", "simple", "complex"]:
    if file_type == "MW":
        mark = "o"
    elif file_type == "simple":
        mark = "s"
    else:
        mark = "^"

    # Filter for mu
    filtered_data_mu = [
        data
        for data in file_data
        # /if data["disc"] == 0 and data["de"] == 0 and data["ga"] == 1 and data["file_type"] == file_type
        if data["disc"] == 0
        and data["de"] == 0
        and data["ga"] == 1
        and data["variant"] == file_type
    ]
    mu_values = []
    avg_d_values_c0_mu = []
    avg_row_averages_mu = []
    avg_row_minima_mu = []
    for data in filtered_data_mu:
        for c, avg_d in data["averages"]:
            if c == 0:
                mu_values.append(data["mu"])
                avg_d_values_c0_mu.append(avg_d)
                avg_row_averages_mu.append(data["avg_row_averages"])
                avg_row_minima_mu.append(data["avg_row_minima"])
                break

    if mu_values:
        # Sort data by mu
        sorted_results_mu = sorted(
            zip(mu_values, avg_d_values_c0_mu, avg_row_averages_mu, avg_row_minima_mu)
        )
        mu_values, avg_d_values_c0_mu, avg_row_averages_mu, avg_row_minima_mu = zip(
            *sorted_results_mu
        )

        # Plot Average d on ax1
        ax1.plot(
            mu_values,
            avg_d_values_c0_mu,
            marker=mark,
            linestyle="-",
            label=f"Average d for c=0 (mu, {file_type})",
            color="blue",
        )

        # Plot satisfaction metrics on ax2
        ax2.plot(
            mu_values,
            avg_row_averages_mu,
            marker=mark,
            linestyle=":",
            label=f"Average Utilitarian Welfare (mu, {file_type})",
            color="blue",
        )
        ax2.plot(
            mu_values,
            avg_row_minima_mu,
            marker=mark,
            linestyle="--",
            label=f"Average Row Minima (mu, {file_type})",
            color="blue",
        )

    # Filter for ga
    filtered_data_ga = [
        data
        for data in file_data
        # /if data["mu"] == 1 and data["de"] == 0 and data["disc"] == 0 and data["file_type"] == file_type
        if data["mu"] == 1
        and data["de"] == 0
        and data["disc"] == 0
        and data["variant"] == file_type
    ]
    ga_values = []
    avg_d_values_c0_ga = []
    avg_row_averages_ga = []
    avg_row_minima_ga = []
    for data in filtered_data_ga:
        for c, avg_d in data["averages"]:
            if c == 0:
                ga_values.append(data["ga"])
                avg_d_values_c0_ga.append(avg_d)
                avg_row_averages_ga.append(data["avg_row_averages"])
                avg_row_minima_ga.append(data["avg_row_minima"])
                break

    print(avg_row_minima_ga)
    if ga_values:
        # Sort data by ga
        sorted_results_ga = sorted(
            zip(ga_values, avg_d_values_c0_ga, avg_row_averages_ga, avg_row_minima_ga)
        )
        ga_values, avg_d_values_c0_ga, avg_row_averages_ga, avg_row_minima_ga = zip(
            *sorted_results_ga
        )

        # Plot Average d on ax1
        ax1.plot(
            ga_values,
            avg_d_values_c0_ga,
            marker=mark,
            linestyle="-",
            label=f"Average d for c=0 (ga, {file_type})",
            color="orange",
        )

        # Plot satisfaction metrics on ax2
        ax2.plot(
            ga_values,
            avg_row_averages_ga,
            marker=mark,
            linestyle=":",
            label=f"Average Utilitarian Welfare (ga, {file_type})",
            color="orange",
        )
        ax2.plot(
            ga_values,
            avg_row_minima_ga,
            marker=mark,
            linestyle="--",
            label=f"Average Row Minima (ga, {file_type})",
            color="orange",
        )

ax1.set_title("Average d vs. mu and ga (Separate by File Type)")
ax1.set_xlabel("Parameter (mu or ga)")
ax1.set_ylabel("Average d (c=0)")
ax1.legend(loc="upper left")
ax1.grid(True)

# Customize the second plot (Satisfaction Metrics)
ax2.set_title("Satisfaction Metrics vs. mu and ga (Separate by File Type)")
ax2.set_xlabel("Parameter (mu or ga)")
ax2.set_ylabel("Satisfaction Metrics")
ax2.legend(loc="upper left")
ax2.grid(True)

# Flip x-axis for both plots
ax1.invert_xaxis()
ax2.invert_xaxis()

# Adjust layout and save
plt.tight_layout()
plt.savefig("gen_separate_plots_average_d_and_satisfaction_metrics.png")
plt.show()
plt.close()


# Create a figure for the combined plot with separate y-axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
# Process DISC: Fix de=0, ga=1, mu=1, vary disc
# /for file_type in ["_random", "_worst"]:
# /    if file_type == "_random":
# /        mark = 'o'
# /    else:
# /        mark = '^'
for file_type in ["MW", "simple", "complex"]:
    if file_type == "MW":
        mark = "o"
    elif file_type == "simple":
        mark = "s"
    else:
        mark = "^"

    filtered_data_disc = [
        data
        for data in file_data
        # /if data["de"] == 0 and data["ga"] == 1 and data["mu"] == 1 and data["file_type"] == file_type
        if data["de"] == 0
        and data["ga"] == 1
        and data["mu"] == 1
        and data["variant"] == file_type
    ]
    disc_values = []
    avg_c_values_d1_disc = []
    avg_row_averages_disc = []
    avg_row_minima_disc = []
    for data in filtered_data_disc:
        # Extract the average c value for d=1
        # for d, avg_c in data["averages_for_d"]:
        # if d == 1:  # Look for d=1
        disc_values.append(data["disc"])
        avg_c_values_d1_disc.append(data["d_viol"])
        avg_row_averages_disc.append(data["avg_row_averages"])
        avg_row_minima_disc.append(data["avg_row_minima"])
        #    break
    if disc_values:
        # Sort data by disc
        sorted_results_disc = sorted(
            zip(
                disc_values,
                avg_c_values_d1_disc,
                avg_row_averages_disc,
                avg_row_minima_disc,
            )
        )
        (
            disc_values,
            avg_c_values_d1_disc,
            avg_row_averages_disc,
            avg_row_minima_disc,
        ) = zip(*sorted_results_disc)

        # Plot Average c for d=1 on ax1
        ax1.plot(
            disc_values,
            avg_c_values_d1_disc,
            marker=mark,
            linestyle="-",
            label=f"Average c for d=1 (disc, {file_type})",
            color="green",
        )

        # Plot satisfaction metrics on ax2
        ax2.plot(
            disc_values,
            avg_row_averages_disc,
            marker=mark,
            linestyle=":",
            label=f"Average Utilitarian Welfare (disc, {file_type})",
            color="green",
        )
        ax2.plot(
            disc_values,
            avg_row_minima_disc,
            marker=mark,
            linestyle="--",
            label=f"Average Row Minima (disc, {file_type})",
            color="green",
        )

# Process DE: Fix disc=0, ga=1, mu=1, vary de
# /for file_type in ["_random", "_worst"]:
# /    if file_type == "_random":
# /        mark = 'o'
# /    else:
# /        mark = '^'

for file_type in ["MW", "simple", "complex"]:
    if file_type == "MW":
        mark = "o"
    elif file_type == "simple":
        mark = "s"
    else:
        mark = "^"
    filtered_data_de = [
        data
        for data in file_data
        # /if data["disc"] == 0 and data["ga"] == 1 and data["mu"] == 1 and data["file_type"] == file_type
        if data["disc"] == 0
        and data["ga"] == 1
        and data["mu"] == 1
        and data["variant"] == file_type
    ]
    de_values = []
    avg_c_values_d1_de = []
    avg_row_averages_de = []
    avg_row_minima_de = []
    for data in filtered_data_de:
        # Extract the average c value for d=1
        # for d, avg_c in data["averages_for_d"]:
        # if d == 1:  # Look for d=1
        de_values.append(data["de"])
        avg_c_values_d1_de.append(data["d_viol"])
        avg_row_averages_de.append(data["avg_row_averages"])
        avg_row_minima_de.append(data["avg_row_minima"])
        #    break
    if de_values:
        # Sort data by de
        sorted_results_de = sorted(
            zip(de_values, avg_c_values_d1_de, avg_row_averages_de, avg_row_minima_de)
        )
        de_values, avg_c_values_d1_de, avg_row_averages_de, avg_row_minima_de = zip(
            *sorted_results_de
        )

        # Plot Average c for d=1 on ax1
        ax1.plot(
            de_values,
            avg_c_values_d1_de,
            marker=mark,
            linestyle="-",
            label=f"Average c for d=1 (de, {file_type})",
            color="purple",
        )

        # Plot satisfaction metrics on ax2
        ax2.plot(
            de_values,
            avg_row_averages_de,
            marker=mark,
            linestyle=":",
            label=f"Average Utilitarian Welfare (de, {file_type})",
            color="purple",
        )
        ax2.plot(
            de_values,
            avg_row_minima_de,
            marker=mark,
            linestyle="--",
            label=f"Average Row Minima (de, {file_type})",
            color="purple",
        )

"""
# Customize axes
ax1.set_title("Average c (d=1) and Satisfaction Metrics vs. disc and de (Separate by File Type)")
ax1.set_xlabel("Parameter (disc or de)")
ax1.set_ylabel("Average c for d=1")
ax1.set_ylim(-1, 5)  # Scale for Average d

ax2.set_ylabel("Satisfaction Metrics (1 to 6)")
ax2.set_ylim(0, 10)  # Scale for satisfaction metrics

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Add grid and finalize
ax1.grid(True)
plt.tight_layout()

# Save and close
plt.savefig("average_c_and_welfare_vs_disc_and_de_fixed.png")
plt.close()
"""


# Customize the first subplot (Average c for d=1)
ax1.set_title("Average c (d=1) vs. disc (Separate by File Type)")
ax1.set_xlabel("disc")
ax1.set_ylabel("Average c for d=1")
# ax1.set_ylim(-1, 5)  # Scale for Average c for d=1
ax1.legend(loc="upper left")
ax1.grid(True)

# Customize the second subplot (Satisfaction Metrics)
ax2.set_title("Satisfaction Metrics vs. disc (Separate by File Type)")
ax2.set_xlabel("disc")
ax2.set_ylabel("Satisfaction Metrics")
# ax2.set_ylim(0, 10)  # Scale for Satisfaction Metrics
ax2.legend(loc="upper left")
ax2.grid(True)

# Adjust layout to ensure there is no overlap
plt.tight_layout()

# Save the figure with side-by-side plots
plt.savefig("separate_plots_average_c_and_satisfaction_metrics.png")

# Display the plot (optional) or close the figure
plt.show()
plt.close()
