import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="Forms_kart College Recommendation Tool",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0;
        opacity: 0.9;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .college-card {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: white;
    }
    .college-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    .chance-high {
        color: #28a745;
        font-weight: bold;
    }
    .chance-good {
        color: #007bff;
        font-weight: bold;
    }
    .chance-fair {
        color: #ffc107;
        font-weight: bold;
    }
    .chance-low {
        color: #dc3545;
        font-weight: bold;
    }
    .stSelectbox > div > div > select {
        background-color: #f8f9fa;
    }
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

class JoSAACounselingTool:
    def __init__(self):
        self.data = None
        self.original_data = None
        
    def load_data(self, uploaded_file):
        """Load data from uploaded Excel file"""
        try:
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Please upload an Excel file (.xlsx or .xls)")
                return False
            
            required_columns = ['Institute', 'Branch', 'Closing_Rank']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Required columns: Institute, Branch, Closing_Rank")
                st.info("Optional columns: Opening Rank, Quota, Seat Type, Gender")
                return False
            
            df = df.dropna(subset=['Institute', 'Branch', 'Closing_Rank'])
            df['Closing_Rank'] = pd.to_numeric(df['Closing_Rank'], errors='coerce')
            df = df.dropna(subset=['Closing_Rank'])
            
            if 'Opening Rank' not in df.columns:
                df['Opening Rank'] = df['Closing_Rank']
            else:
                df['Opening Rank'] = pd.to_numeric(df['Opening Rank'], errors='coerce')
                df['Opening Rank'] = df['Opening Rank'].fillna(df['Closing_Rank'])
            
            if 'Quota' not in df.columns:
                df['Quota'] = 'OS'
            
            if 'Seat Type' not in df.columns:
                df['Seat Type'] = 'General'
                
            if 'Gender' not in df.columns:
                df['Gender'] = 'Gender-Neutral'
            
            self.data = df
            self.original_data = df.copy()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return False
    
    def get_unique_values(self, column):
        """Get unique values from a column"""
        if self.data is not None and column in self.data.columns:
            return sorted(self.data[column].unique().tolist())
        return []
    
    def extract_branch_name(self, full_branch):
        """Extract main branch name from full description"""
        if pd.isna(full_branch):
            return ""
        branch_str = str(full_branch)
        if '(' in branch_str:
            return branch_str.split('(')[0].strip()
        return branch_str.strip()
    
    def get_chance_level(self, student_rank, closing_rank, opening_rank):
        """Determine admission chance level"""
        try:
            rank = int(student_rank)
            closing = int(closing_rank)
            opening = int(opening_rank) if pd.notna(opening_rank) else closing
            
            if rank <= opening:
                return "High", "chance-high"
            elif rank <= (opening + closing) / 2:
                return "Good", "chance-good"
            elif rank <= closing:
                return "Fair", "chance-fair"
            else:
                return "Low", "chance-low"
        except:
            return "Unknown", "chance-low"
    
    # --- Modified: Reverted to single branch parameter for individual searches ---
    def search_colleges(self, student_rank, branch=None, quota=None, seat_type=None, gender=None, top_n=5):
        """Search for suitable colleges based on criteria"""
        if self.data is None:
            return pd.DataFrame()
        
        try:
            rank = int(student_rank)
        except:
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        
        filtered_data = filtered_data[filtered_data['Closing_Rank'] >= rank]
        
        if branch:
            filtered_data = filtered_data[filtered_data['Branch'] == branch]
        
        if quota:
            filtered_data = filtered_data[filtered_data['Quota'] == quota]
            
        if seat_type:
            filtered_data = filtered_data[filtered_data['Seat Type'] == seat_type]
            
        if gender:
            filtered_data = filtered_data[filtered_data['Gender'] == gender]
        
        filtered_data = filtered_data.sort_values('Closing_Rank')
        
        return filtered_data.head(top_n)
    
    def get_statistics(self):
        """Get data statistics"""
        if self.data is None:
            return {}
        
        stats = {
            'total_records': len(self.data),
            'unique_institutes': self.data['Institute'].nunique(),
            'unique_branches': self.data['Branch'].nunique(),
            'rank_range': {
                'min': int(self.data['Closing_Rank'].min()),
                'max': int(self.data['Closing_Rank'].max())
            }
        }
        return stats

def main():
    if 'counseling_tool' not in st.session_state:
        st.session_state.counseling_tool = JoSAACounselingTool()
    
    tool = st.session_state.counseling_tool
    
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Forms_kart College Recommendation Tool</h1>
        <p>Find the best colleges based on your JEE rank and preferences</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Excel File",
            type=['xlsx', 'xls'],
            help="Upload Excel file with college data. Required columns: Institute, Branch, Closing_Rank"
        )
        
        if uploaded_file is not None:
            if st.button("Load Data"):
                with st.spinner("Loading data..."):
                    if tool.load_data(uploaded_file):
                        st.success("✅ Data loaded successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to load data")
        
        if tool.data is not None:
            st.header("📊 Data Overview")
            stats = tool.get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", stats['total_records'])
                st.metric("Unique Branches", stats['unique_branches'])
            
            with col2:
                st.metric("Unique Institutes", stats['unique_institutes'])
                st.metric("Rank Range", f"{stats['rank_range']['min']:,} - {stats['rank_range']['max']:,}")
    
    if tool.data is None:
        st.info("👆 Please upload an Excel file to get started")
        
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'Institute': ['NIT Surathkal', 'NIT Rourkela', 'IIIT Hyderabad'],
            'Branch': ['Computer Science Engineering', 'Electrical Engineering', 'Electronics Engineering'],
            'Closing_Rank': [1250, 2100, 1800],
            'Opening Rank': [890, 1650, 1200],
            'Quota': ['OS', 'OS', 'OS'],
            'Seat Type': ['General', 'General', 'EWS'],
            'Gender': ['Gender-Neutral', 'Gender-Neutral', 'Gender-Neutral']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("""
        **Required Columns:**
        - `Institute`: Name of the college/institute
        - `Branch`: Name of the branch/course
        - `Closing_Rank`: Last rank that got admission
        
        **Optional Columns:**
        - `Opening Rank`: First rank that got admission
        - `Quota`: OS (Other State), HS (Home State), etc.
        - `Seat Type`: General, EWS, OBC, SC, ST, etc.
        - `Gender`: Gender-Neutral, Female-only, etc.
        """)
        
    else:
        st.header("🔍 Find Your Best Colleges")
        
        # --- Modified: Four columns for rank and three branch preferences ---
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            student_rank = st.number_input(
                "Your JEE Rank *",
                min_value=1,
                max_value=1000000,
                value=None,
                placeholder="Enter your rank",
                help="Enter your JEE Main/Advanced rank"
            )
        
        branches = tool.get_unique_values('Branch')
        default_option = ["No Preference"]
        
        with col2:
            branch_1 = st.selectbox(
                "First Preference Branch",
                default_option + branches,
                index=0,
                help="Select your first preferred branch"
            )
        
        with col3:
            branch_2 = st.selectbox(
                "Second Preference Branch",
                default_option + branches,
                index=0,
                help="Select your second preferred branch"
            )
        
        with col4:
            branch_3 = st.selectbox(
                "Third Preference Branch",
                default_option + branches,
                index=0,
                help="Select your third preferred branch"
            )
        
        # Number of results
        top_n = st.selectbox(
            "Number of Results",
            [5, 10, 15, 20],
            index=0,
            help="How many college recommendations to show per branch"
        )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters (Optional)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                quotas = ['All Quotas'] + tool.get_unique_values('Quota')
                selected_quota = st.selectbox("Quota", quotas)
                if selected_quota == 'All Quotas':
                    selected_quota = None
            
            with col2:
                seat_types = ['All Seat Types'] + tool.get_unique_values('Seat Type')
                selected_seat_type = st.selectbox("Seat Type", seat_types)
                if selected_seat_type == 'All Seat Types':
                    selected_seat_type = None
            
            with col3:
                genders = ['All'] + tool.get_unique_values('Gender')
                selected_gender = st.selectbox("Gender", genders)
                if selected_gender == 'All':
                    selected_gender = None
        
        # --- Modified: Separate searches for each branch preference ---
        if st.button("🚀 Find Best Colleges", type="primary", use_container_width=True):
            if student_rank is None:
                st.error("⚠️ Please enter your JEE rank")
            elif branch_1 == "No Preference" and branch_2 == "No Preference" and branch_3 == "No Preference":
                st.error("⚠️ Please select at least one branch preference")
            else:
                with st.spinner("Searching for best colleges..."):
                    # List of branch preferences with labels
                    preferences = [
                        ("First Preference Branch", branch_1),
                        ("Second Preference Branch", branch_2),
                        ("Third Preference Branch", branch_3)
                    ]
                    
                    for pref_name, selected_branch in preferences:
                        if selected_branch != "No Preference":
                            st.subheader(f"Results for {pref_name}: {selected_branch}")
                            # Perform search for this specific branch
                            results = tool.search_colleges(
                                student_rank,
                                selected_branch,
                                selected_quota,
                                selected_seat_type,
                                selected_gender,
                                top_n
                            )
                            
                            if len(results) > 0:
                                st.success(f"🎉 Found {len(results)} suitable colleges for rank {student_rank:,} and branch {selected_branch}")
                                
                                # Display college cards
                                for idx, (_, college) in enumerate(results.iterrows(), 1):
                                    chance_level, chance_class = tool.get_chance_level(
                                        student_rank, college['Closing_Rank'], college['Opening Rank']
                                    )
                                    
                                    with st.container():
                                        st.markdown(f"""
                                        <div class="college-card">
                                            <div style="display: flex; justify-content: between; align-items: center;">
                                                <h3 style="color: #667eea; margin: 0;">
                                                    #{idx} {college['Institute']}
                                                </h3>
                                                <span class="{chance_class}" style="font-size: 14px;">
                                                    {chance_level} Chance
                                                </span>
                                            </div>
                                            <p style="margin: 8px 0; color: #666;">
                                                <strong>Branch:</strong> {tool.extract_branch_name(college['Branch'])}
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        
                                        with col1:
                                            st.metric("Closing Rank", f"{int(college['Closing_Rank']):,}")
                                        
                                        with col2:
                                            opening_rank = int(college['Opening Rank']) if pd.notna(college['Opening Rank']) else int(college['Closing_Rank'])
                                            st.metric("Opening Rank", f"{opening_rank:,}")
                                        
                                        with col3:
                                            st.metric("Quota", college.get('Quota', 'OS'))
                                        
                                        with col4:
                                            st.metric("Seat Type", college.get('Seat Type', 'General'))
                                        
                                        st.markdown("---")
                                
                                # Visualization
                                st.subheader(f"📈 Rank Analysis for {pref_name}")
                                
                                chart_data = []
                                for _, college in results.iterrows():
                                    chart_data.append({
                                        'College': college['Institute'][:30] + "..." if len(college['Institute']) > 30 else college['Institute'],
                                        'Your Rank': student_rank,
                                        'Opening Rank': int(college['Opening Rank']) if pd.notna(college['Opening Rank']) else int(college['Closing_Rank']),
                                        'Closing Rank': int(college['Closing_Rank'])
                                    })
                                
                                chart_df = pd.DataFrame(chart_data)
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Bar(
                                    name='Closing Rank',
                                    x=chart_df['College'],
                                    y=chart_df['Closing Rank'],
                                    marker_color='lightcoral'
                                ))
                                
                                fig.add_trace(go.Bar(
                                    name='Opening Rank',
                                    x=chart_df['College'],
                                    y=chart_df['Opening Rank'],
                                    marker_color='lightblue'
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    name='Your Rank',
                                    x=chart_df['College'],
                                    y=[student_rank] * len(chart_df),
                                    mode='lines+markers',
                                    line=dict(color='red', width=3, dash='dash'),
                                    marker=dict(size=8)
                                ))
                                
                                fig.update_layout(
                                    title=f"Rank Comparison for {pref_name}",
                                    xaxis_title="Colleges",
                                    yaxis_title="Rank",
                                    barmode='group',
                                    height=500,
                                    showlegend=True
                                )
                                
                                # --- Retained Fix: Use update_xaxes instead of update_xaxis ---
                                fig.update_xaxes(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download results
                                st.subheader(f"💾 Download Results for {pref_name}")
                                
                                download_data = results.copy()
                                download_data['Your_Rank'] = student_rank
                                download_data['Branch_Name'] = download_data['Branch'].apply(tool.extract_branch_name)
                                
                                download_data['Admission_Chance'] = download_data.apply(
                                    lambda row: tool.get_chance_level(student_rank, row['Closing_Rank'], row['Opening Rank'])[0],
                                    axis=1
                                )
                                
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    download_data.to_excel(writer, sheet_name='Recommended_Colleges', index=False)
                                
                                st.download_button(
                                    label=f"📥 Download Results for {pref_name} as Excel",
                                    data=output.getvalue(),
                                    file_name=f"josaa_recommendations_rank_{student_rank}_{pref_name.replace(' ', '_').lower()}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                                
                            else:
                                st.warning(f"😔 No colleges found for {pref_name}: {selected_branch}. Try:")
                                st.markdown("""
                                - Increasing your rank range
                                - Adjusting advanced filters
                                - Checking if your rank is within the dataset range
                                """)
                            st.markdown("---")
        
        with st.expander("👀 Preview Loaded Data"):
            st.dataframe(tool.data.head(10), use_container_width=True)

if __name__ == "__main__":
    main()