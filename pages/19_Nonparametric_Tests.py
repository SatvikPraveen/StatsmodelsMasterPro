# 19_Nonparametric_Tests.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# -----------------------------------------------
# 🚀 Page Config
# -----------------------------------------------
st.set_page_config(
    page_title="Nonparametric Tests – StatsmodelsMasterPro",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Nonparametric Statistical Tests")
st.markdown("""
When parametric assumptions fail (non-normal distributions, ordinal data), use **nonparametric tests**:
- **Mann-Whitney U Test** - Two independent samples (alternative to t-test)
- **Wilcoxon Signed-Rank** - Paired samples (alternative to paired t-test)
- **Kruskal-Wallis H Test** - Multiple groups (alternative to ANOVA)
- **Friedman Test** - Repeated measures (alternative to repeated measures ANOVA)
""")

# -----------------------------------------------
# 📥 Load Data
# -----------------------------------------------
DATA_PATH = Path(__file__).parent.parent / "synthetic_data" / "posthoc_dataset.csv"
df = pd.read_csv(DATA_PATH)

st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("📏 Total Observations", df.shape[0])
with col2:
    st.metric("📊 Unique Groups", df['Group'].nunique())

# -----------------------------------------------
# 📊 Select Test Type
# -----------------------------------------------
test_type = st.sidebar.selectbox(
    "🔧 Select Nonparametric Test",
    [
        "Mann-Whitney U Test (2 groups)",
        "Wilcoxon Signed-Rank Test (paired)",
        "Kruskal-Wallis H Test (3+ groups)",
        "Friedman Test (repeated measures)"
    ]
)

# ===============================================
# 🔹 MANN-WHITNEY U TEST
# ===============================================
if test_type == "Mann-Whitney U Test (2 groups)":
    st.header("🔹 Mann-Whitney U Test")
    
    st.markdown("""
    **Mann-Whitney U Test** (also called Wilcoxon rank-sum test) is the **nonparametric alternative to the independent t-test**.
    
    - **Use when:** Data is ordinal or non-normally distributed
    - **Null Hypothesis:** The distributions of both groups are equal
    - **Test:** Compares median ranks, not means
    """)
    
    # Select groups
    groups = df['Group'].unique()
    
    if len(groups) < 2:
        st.error("❌ Need at least 2 groups for Mann-Whitney U test.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            group1_name = st.selectbox("Select Group 1", groups, index=0)
        with col2:
            group2_name = st.selectbox("Select Group 2", [g for g in groups if g != group1_name], index=0)
        
        response_var = st.selectbox("Select Response Variable", ['Score'])
        
        if st.button("🔍 Run Mann-Whitney U Test"):
            # Extract data
            group1_data = df[df['Group'] == group1_name][response_var].dropna()
            group2_data = df[df['Group'] == group2_name][response_var].dropna()
            
            # Run test
            statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            st.success("✅ Test complete!")
            
            # Results
            st.subheader("📊 Test Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("U Statistic", f"{statistic:.2f}")
            col2.metric("p-value", f"{p_value:.4f}")
            col3.metric("Significant (α=0.05)", "✅ Yes" if p_value < 0.05 else "❌ No")
            
            # Descriptive stats
            st.subheader("📈 Descriptive Statistics")
            desc_df = pd.DataFrame({
                'Group': [group1_name, group2_name],
                'N': [len(group1_data), len(group2_data)],
                'Median': [group1_data.median(), group2_data.median()],
                'Mean Rank': [group1_data.rank().mean(), group2_data.rank().mean()],
                'Min': [group1_data.min(), group2_data.min()],
                'Max': [group1_data.max(), group2_data.max()]
            })
            st.dataframe(desc_df.style.format(precision=2))
            
            # Visualization
            st.subheader("📊 Distribution Comparison")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Box plot
            axes[0].boxplot([group1_data, group2_data], labels=[group1_name, group2_name])
            axes[0].set_ylabel(response_var)
            axes[0].set_title('Box Plot Comparison')
            axes[0].grid(alpha=0.3)
            
            # Violin plot
            combined_df = pd.concat([
                pd.DataFrame({response_var: group1_data, 'Group': group1_name}),
                pd.DataFrame({response_var: group2_data, 'Group': group2_name})
            ])
            
            parts = axes[1].violinplot([group1_data, group2_data], positions=[1, 2], 
                                        showmeans=True, showmedians=True)
            axes[1].set_xticks([1, 2])
            axes[1].set_xticklabels([group1_name, group2_name])
            axes[1].set_ylabel(response_var)
            axes[1].set_title('Violin Plot Comparison')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Interpretation
            st.subheader("📖 Interpretation")
            if p_value < 0.05:
                st.success(f"✅ **Significant difference** found between {group1_name} and {group2_name} (p < 0.05)")
                st.markdown(f"The median of **{group1_name}** ({group1_data.median():.2f}) differs significantly from **{group2_name}** ({group2_data.median():.2f}).")
            else:
                st.info(f"ℹ️ **No significant difference** found between {group1_name} and {group2_name} (p ≥ 0.05)")

# ===============================================
# 🔹 WILCOXON SIGNED-RANK TEST
# ===============================================
elif test_type == "Wilcoxon Signed-Rank Test (paired)":
    st.header("🔹 Wilcoxon Signed-Rank Test")
    
    st.markdown("""
    **Wilcoxon Signed-Rank Test** is the **nonparametric alternative to the paired t-test**.
    
    - **Use when:** Paired data is ordinal or non-normally distributed
    - **Null Hypothesis:** Median difference between pairs is zero
    - **Example:** Before/after measurements on same subjects
    """)
    
    st.info("💡 **Demo:** We'll generate paired data (before/after treatment) for illustration.")
    
    # Generate paired data
    n_pairs = st.slider("Number of Paired Observations", 20, 200, 50)
    
    if st.button("🎲 Generate Paired Data & Run Test"):
        np.random.seed(42)
        
        # Simulate before/after data
        before = np.random.normal(70, 10, n_pairs)
        treatment_effect = np.random.normal(5, 3, n_pairs)  # Variable treatment effect
        after = before + treatment_effect
        
        paired_df = pd.DataFrame({
            'Before': before,
            'After': after,
            'Difference': after - before
        })
        
        st.subheader("📊 Paired Data Preview")
        st.dataframe(paired_df.head(10))
        
        # Run Wilcoxon test
        statistic, p_value = stats.wilcoxon(paired_df['Before'], paired_df['After'])
        
        st.success("✅ Test complete!")
        
        # Results
        st.subheader("📊 Test Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Wilcoxon Statistic", f"{statistic:.2f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Significant (α=0.05)", "✅ Yes" if p_value < 0.05 else "❌ No")
        
        # Descriptive stats
        st.subheader("📈 Descriptive Statistics")
        desc_df = pd.DataFrame({
            'Measure': ['Before', 'After', 'Difference'],
            'Median': [paired_df['Before'].median(), paired_df['After'].median(), paired_df['Difference'].median()],
            'Mean': [paired_df['Before'].mean(), paired_df['After'].mean(), paired_df['Difference'].mean()],
            'Std': [paired_df['Before'].std(), paired_df['After'].std(), paired_df['Difference'].std()]
        })
        st.dataframe(desc_df.style.format(precision=2))
        
        # Visualization
        st.subheader("📊 Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Before vs After
        axes[0].scatter(paired_df['Before'], paired_df['After'], alpha=0.6, edgecolors='k', linewidth=0.5)
        axes[0].plot([paired_df['Before'].min(), paired_df['Before'].max()], 
                      [paired_df['Before'].min(), paired_df['Before'].max()], 
                      'r--', linewidth=2, label='No change line')
        axes[0].set_xlabel('Before')
        axes[0].set_ylabel('After')
        axes[0].set_title('Before vs After Scores')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Difference distribution
        axes[1].hist(paired_df['Difference'], bins=20, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        axes[1].axvline(paired_df['Difference'].median(), color='green', linestyle='--', linewidth=2, label='Median difference')
        axes[1].set_xlabel('Difference (After - Before)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Differences')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        st.subheader("📖 Interpretation")
        if p_value < 0.05:
            st.success(f"✅ **Significant change** detected (p < 0.05)")
            st.markdown(f"Median difference: {paired_df['Difference'].median():.2f}")
        else:
            st.info(f"ℹ️ **No significant change** detected (p ≥ 0.05)")

# ===============================================
# 🔹 KRUSKAL-WALLIS H TEST
# ===============================================
elif test_type == "Kruskal-Wallis H Test (3+ groups)":
    st.header("🔹 Kruskal-Wallis H Test")
    
    st.markdown("""
    **Kruskal-Wallis H Test** is the **nonparametric alternative to one-way ANOVA**.
    
    - **Use when:** Comparing 3+ independent groups with non-normal data
    - **Null Hypothesis:** All groups have the same distribution
    - **Test:** Compares median ranks across groups
    """)
    
    response_var = st.selectbox("Select Response Variable", ['Score'])
    group_var = 'Group'
    
    if st.button("🔍 Run Kruskal-Wallis Test"):
        # Extract groups
        groups_data = [df[df[group_var] == group][response_var].dropna() for group in df[group_var].unique()]
        
        # Run test
        statistic, p_value = stats.kruskal(*groups_data)
        
        st.success("✅ Test complete!")
        
        # Results
        st.subheader("📊 Test Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("H Statistic", f"{statistic:.4f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Groups", len(groups_data))
        col4.metric("Significant (α=0.05)", "✅ Yes" if p_value < 0.05 else "❌ No")
        
        # Descriptive stats by group
        st.subheader("📈 Group Statistics")
        group_stats = []
        for group_name in df[group_var].unique():
            group_data = df[df[group_var] == group_name][response_var]
            group_stats.append({
                'Group': group_name,
                'N': len(group_data),
                'Median': group_data.median(),
                'Mean': group_data.mean(),
                'Mean Rank': group_data.rank().mean(),
                'IQR': group_data.quantile(0.75) - group_data.quantile(0.25)
            })
        
        stats_df = pd.DataFrame(group_stats)
        st.dataframe(stats_df.style.format(precision=2))
        
        # Visualization
        st.subheader("📊 Distribution by Group")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Box plot
        df.boxplot(column=response_var, by=group_var, ax=axes[0])
        axes[0].set_xlabel('Group')
        axes[0].set_ylabel(response_var)
        axes[0].set_title('Box Plot by Group')
        axes[0].get_figure().suptitle('')  # Remove automatic title
        
        # Violin plot
        parts = axes[1].violinplot([df[df[group_var] == g][response_var] for g in df[group_var].unique()],
                                    positions=range(len(df[group_var].unique())),
                                    showmeans=True, showmedians=True)
        axes[1].set_xticks(range(len(df[group_var].unique())))
        axes[1].set_xticklabels(df[group_var].unique())
        axes[1].set_xlabel('Group')
        axes[1].set_ylabel(response_var)
        axes[1].set_title('Violin Plot by Group')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        st.subheader("📖 Interpretation")
        if p_value < 0.05:
            st.success("✅ **Significant difference** found between groups (p < 0.05)")
            st.markdown("At least one group differs significantly from the others.")
            st.info("💡 Consider **post-hoc pairwise comparisons** (e.g., Dunn's test) to identify which groups differ.")
        else:
            st.info("ℹ️ **No significant difference** found between groups (p ≥ 0.05)")

# ===============================================
# 🔹 FRIEDMAN TEST
# ===============================================
elif test_type == "Friedman Test (repeated measures)":
    st.header("🔹 Friedman Test")
    
    st.markdown("""
    **Friedman Test** is the **nonparametric alternative to repeated measures ANOVA**.
    
    - **Use when:** Multiple related/matched groups with non-normal data
    - **Null Hypothesis:** Distributions are the same across repeated measures
    - **Example:** Same subjects measured at different time points
    """)
    
    st.info("💡 **Demo:** We'll generate repeated measures data for illustration.")
    
    n_subjects = st.slider("Number of Subjects", 10, 100, 30)
    n_timepoints = st.slider("Number of Time Points", 3, 6, 4)
    
    if st.button("🎲 Generate Repeated Measures Data & Run Test"):
        np.random.seed(42)
        
        # Generate repeated measures data
        data_list = []
        for subject in range(n_subjects):
            baseline = np.random.normal(70, 10)
            measurements = [baseline + np.random.normal(i*2, 5) for i in range(n_timepoints)]
            data_list.append(measurements)
        
        data_array = np.array(data_list)
        
        # Convert to DataFrame for display
        friedman_df = pd.DataFrame(data_array, 
                                     columns=[f'Time_{i+1}' for i in range(n_timepoints)])
        friedman_df.insert(0, 'Subject', [f'S{i+1}' for i in range(n_subjects)])
        
        st.subheader("📊 Repeated Measures Data Preview")
        st.dataframe(friedman_df.head(10))
        
        # Run Friedman test
        statistic, p_value = stats.friedmanchisquare(*data_array.T)
        
        st.success("✅ Test complete!")
        
        # Results
        st.subheader("📊 Test Results")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chi-square Statistic", f"{statistic:.4f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Time Points", n_timepoints)
        col4.metric("Significant (α=0.05)", "✅ Yes" if p_value < 0.05 else "❌ No")
        
        # Descriptive stats by time point
        st.subheader("📈 Time Point Statistics")
        time_stats = []
        for i in range(n_timepoints):
            time_stats.append({
                'Time Point': f'Time_{i+1}',
                'Median': np.median(data_array[:, i]),
                'Mean': np.mean(data_array[:, i]),
                'Std': np.std(data_array[:, i])
            })
        
        stats_df = pd.DataFrame(time_stats)
        st.dataframe(stats_df.style.format(precision=2))
        
        # Visualization
        st.subheader("📊 Visualization")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Box plot
        axes[0].boxplot(data_array.T, labels=[f'T{i+1}' for i in range(n_timepoints)])
        axes[0].set_xlabel('Time Point')
        axes[0].set_ylabel('Measurement')
        axes[0].set_title('Box Plot Across Time Points')
        axes[0].grid(alpha=0.3)
        
        # Line plot showing trends
        for i in range(min(10, n_subjects)):  # Plot first 10 subjects
            axes[1].plot(range(1, n_timepoints+1), data_array[i, :], 'o-', alpha=0.5, linewidth=1)
        
        means = data_array.mean(axis=0)
        axes[1].plot(range(1, n_timepoints+1), means, 'ro-', linewidth=3, markersize=10, label='Mean')
        axes[1].set_xlabel('Time Point')
        axes[1].set_ylabel('Measurement')
        axes[1].set_title('Individual Trajectories (first 10 subjects)')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        axes[1].set_xticks(range(1, n_timepoints+1))
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Interpretation
        st.subheader("📖 Interpretation")
        if p_value < 0.05:
            st.success("✅ **Significant difference** across time points (p < 0.05)")
            st.markdown("The measurements show significant variation across repeated time points.")
        else:
            st.info("ℹ️ **No significant difference** across time points (p ≥ 0.05)")

# -----------------------------------------------
# 📚 Comparison Table
# -----------------------------------------------
st.markdown("---")
st.subheader("📚 Nonparametric vs Parametric Tests")

comparison_df = pd.DataFrame({
    'Parametric Test': [
        'Independent t-test',
        'Paired t-test',
        'One-way ANOVA',
        'Repeated measures ANOVA'
    ],
    'Nonparametric Alternative': [
        'Mann-Whitney U',
        'Wilcoxon Signed-Rank',
        'Kruskal-Wallis H',
        'Friedman Test'
    ],
    'Use Case': [
        '2 independent groups',
        'Paired/matched samples',
        '3+ independent groups',
        'Repeated measures'
    ],
    'Assumption': [
        'Normality not required',
        'Normality not required',
        'Normality not required',
        'Normality not required'
    ]
})

st.dataframe(comparison_df, use_container_width=True)

# -----------------------------------------------
# 🧾 Footer
# -----------------------------------------------
st.markdown("---")
st.caption("StatsmodelsMasterPro • Streamlit App • Nonparametric Tests Module")
