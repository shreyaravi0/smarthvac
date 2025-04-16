import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pytz

# Constants
WEATHER_API_KEY = "7a93a3c2d66e9fee6a9a03ff2170ba7e"  # Replace with your actual key
ZONES = ["Kitchen", "Bathroom", "Bedroom"]
INDIAN_CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", 
                "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

ZONE_MODES = {
    "COMFORT": {"temp_range": (20.0, 24.0), "energy_factor": 1.2},
    "ECO": {"temp_range": (18.0, 26.0), "energy_factor": 0.8},
    "AUTO": {"temp_range": (19.0, 25.0), "energy_factor": 1.0}
}

BADGES = {
    0: "🌱 Beginner Eco-Warrior",
    50: "🌿 Green Guardian",
    100: "🌳 Eco Hero",
    200: "🌎 Sustainability Champion"
}

class Alert:
    def __init__(self):
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'energy_spikes' not in st.session_state:
            st.session_state.energy_spikes = [np.random.choice([True, False], p=[0.3, 0.7]) for _ in range(3)]

    def check_alerts(self, zones_data):
        current_alerts = []
        for i, usage in enumerate(zones_data['energy_usage']):
            if usage > 350 or st.session_state.energy_spikes[i]:
                current_alerts.append({
                    "zone": ZONES[i],
                    "message": f"⚠️ High energy usage in {ZONES[i]}: {usage:.1f} kWh",
                    "suggestions": [
                        "Switch to ECO mode",
                        "Optimize airflow settings",
                        f"Current settings causing {np.random.randint(15, 30)}% higher usage"
                    ]
                })
        st.session_state.alerts = current_alerts

    def display_alerts(self):
        if st.session_state.alerts:
            for alert in st.session_state.alerts:
                with st.expander(alert["message"]):
                    for suggestion in alert["suggestions"]:
                        st.write(f"- {suggestion}")
                    if st.button(f"Switch {alert['zone']} to ECO mode"):
                        st.success(f"Switched {alert['zone']} to ECO mode")
        else:
            st.success("All systems operating normally")

class WeatherService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.weatherstack.com/current"
    
    def get_weather(self, city):
        try:
            params = {
                'access_key': self.api_key,
                'query': city,
                'units': 'm'
            }
            response = requests.get(self.base_url, params=params)
            data = response.json()
            if 'current' in data:
                return {
                    'temperature': data['current']['temperature'],
                    'humidity': data['current']['humidity'],
                    'condition': data['current']['weather_descriptions'][0],
                    'wind_speed': data['current']['wind_speed'],
                    'feels_like': data['current']['feelslike'],
                    'pressure': data['current']['pressure']
                }
        except:
            pass
        
        # Mock data for different cities to ensure changes are visible when city is changed
        mock_data = {
            "Mumbai": {'temperature': 32, 'humidity': 78, 'condition': 'Partly cloudy', 'wind_speed': 5, 'feels_like': 36, 'pressure': 1012},
            "Delhi": {'temperature': 28, 'humidity': 55, 'condition': 'Sunny', 'wind_speed': 8, 'feels_like': 30, 'pressure': 1010},
            "Bangalore": {'temperature': 24, 'humidity': 65, 'condition': 'Light rain', 'wind_speed': 3, 'feels_like': 25, 'pressure': 1015},
            "Hyderabad": {'temperature': 29, 'humidity': 60, 'condition': 'Clear', 'wind_speed': 4, 'feels_like': 31, 'pressure': 1014},
            "Chennai": {'temperature': 31, 'humidity': 75, 'condition': 'Humid', 'wind_speed': 6, 'feels_like': 34, 'pressure': 1011},
            "Kolkata": {'temperature': 30, 'humidity': 80, 'condition': 'Overcast', 'wind_speed': 7, 'feels_like': 33, 'pressure': 1009},
            "Pune": {'temperature': 26, 'humidity': 50, 'condition': 'Clear', 'wind_speed': 4, 'feels_like': 27, 'pressure': 1013},
            "Ahmedabad": {'temperature': 33, 'humidity': 45, 'condition': 'Sunny', 'wind_speed': 9, 'feels_like': 35, 'pressure': 1008},
            "Jaipur": {'temperature': 31, 'humidity': 40, 'condition': 'Hot', 'wind_speed': 10, 'feels_like': 33, 'pressure': 1007},
            "Lucknow": {'temperature': 27, 'humidity': 60, 'condition': 'Cloudy', 'wind_speed': 5, 'feels_like': 29, 'pressure': 1012}
        }
        
        return mock_data.get(city, {'temperature': 25, 'humidity': 65, 'condition': 'Clear', 'wind_speed': 3, 'feels_like': 27, 'pressure': 1013})

class HVACPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self._train_model()
    
    def _train_model(self):
        n_samples = 1000
        np.random.seed(42)
        X = np.random.rand(n_samples, 6)
        X[:, 0] = X[:, 0] * 20 + 10
        X[:, 1] = np.random.randint(0, 30, n_samples)
        X[:, 2] = np.random.randint(0, 24, n_samples)
        X[:, 3] = np.random.randint(0, 7, n_samples)
        X[:, 4] = np.random.uniform(30, 70, n_samples)
        X[:, 5] = np.random.uniform(0, 10, n_samples)
        y = 22 + (X[:, 0] - 20) * 0.3
        y += (X[:, 1] / 30) * 2
        y += np.sin(X[:, 2] * np.pi / 12) * 0.5
        y -= (X[:, 4] - 50) * 0.02
        y -= X[:, 5] * 0.1
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
    
    def predict_optimal_temperature(self, weather_data, occupancy, time_of_day, day_of_week):
        features = np.array([[weather_data['temperature'], occupancy, time_of_day, day_of_week, weather_data['humidity'], weather_data['wind_speed']]])
        features_scaled = self.scaler.transform(features)
        return max(18, min(26, self.model.predict(features_scaled)[0]))

def generate_energy_data_with_weather(temp, humidity, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.datetime.now(ist)
    dates = pd.date_range(end=current_time, periods=24, freq='H')
    base_usage = 250
    humidity_factor = 1 + (humidity - 50) * 0.01
    temp_factor = 1 + abs(temp - 22) * 0.05
    time_factors = np.array([
        0.7, 0.6, 0.5, 0.4, 0.4, 0.5,
        0.7, 1.0, 1.2, 1.1, 1.0,
        1.1, 1.2, 1.1, 1.0, 1.1,
        1.2, 1.3, 1.4, 1.3, 1.2,
        1.1, 0.9, 0.8
    ])
    usage = base_usage * time_factors * humidity_factor * temp_factor + np.random.normal(0, 20, 24)
    return pd.DataFrame({'Time': dates, 'Usage': usage})

def plot_energy_usage(data, zone_name=""):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=data, x='Time', y='Usage', color='#00ff00', linewidth=2)
    ax.set_facecolor('#1E1E2F')
    fig.patch.set_facecolor('#1E1E2F')
    ax.set_xlabel('Time', color='white', fontsize=12)
    ax.set_ylabel('Energy Usage (kWh)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.3)
    title = f'{zone_name} 24-Hour Energy Usage Overview' if zone_name else '24-Hour Energy Usage Overview'
    plt.title(title, color='white', fontsize=14, pad=20)
    plt.tight_layout()
    return fig

def custom_css():
    return """
    <style>
    .stApp {
        background-color: #1E1E2F;
        color: white;
    }
    .control-card, .weather-card {
        background-color: #2C2C44;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    .section-header {
        color: #00ff00;
        font-size: 24px;
        margin: 20px 0;
        font-weight: bold;
    }
    </style>
    """

def initialize_session_state():
    if "zones_data" not in st.session_state:
        st.session_state.zones_data = {
            "temperature": [22.0, 22.0, 22.0],
            "air_quality": [75.0, 75.0, 75.0],
            "energy_usage": [250.0, 250.0, 250.0],
            "occupancy": [10, 10, 10],
            "humidity": [50.0, 50.0, 50.0],
            "mode": ["AUTO", "AUTO", "AUTO"],
            "airflow": ["Auto", "Auto", "Auto"]
        }
    if "eco_points" not in st.session_state:
        st.session_state.eco_points = 0
    if "current_city" not in st.session_state:
        st.session_state.current_city = ""

def update_points(mode):
    if mode == "ECO":
        st.session_state.eco_points += 10

def update_zone_based_on_weather(zone_index, weather_data, predictor):
    """Update zone settings based on weather data"""
    # Update only if in AUTO mode
    if st.session_state.zones_data["mode"][zone_index] == "AUTO":
        optimal_temp = predictor.predict_optimal_temperature(
            weather_data, 
            st.session_state.zones_data["occupancy"][zone_index],
            datetime.datetime.now().hour,
            datetime.datetime.now().weekday()
        )
        st.session_state.zones_data["temperature"][zone_index] = optimal_temp
        st.session_state.zones_data["energy_usage"][zone_index] = optimal_temp * 10 * (1 + (weather_data['humidity'] - 50) * 0.005)
    
    # Update humidity based on weather
    st.session_state.zones_data["humidity"][zone_index] = weather_data['humidity'] + (zone_index - 1) * 2

def display_zone_controls(zone_index, weather_data, predictor):
    zone = ZONES[zone_index]
    st.markdown(f"""<div class='control-card'><h2>{zone} Control Panel</h2></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Status")
        st.metric("Temperature", f"{st.session_state.zones_data['temperature'][zone_index]:.1f}°C")
        st.metric("Air Quality", f"{st.session_state.zones_data['air_quality'][zone_index]}")
        st.metric("Energy Usage", f"{st.session_state.zones_data['energy_usage'][zone_index]:.1f} kWh")
        st.metric("Humidity", f"{st.session_state.zones_data['humidity'][zone_index]:.1f}%")
    with col2:
        st.subheader("Controls")
        mode = st.selectbox("Zone Mode", list(ZONE_MODES.keys()), index=list(ZONE_MODES.keys()).index(st.session_state.zones_data["mode"][zone_index]), key=f"mode_{zone_index}")
        settings = ZONE_MODES[mode]
        st.session_state.zones_data["mode"][zone_index] = mode
        update_points(mode)
        
        if mode != "AUTO":
            min_temp, max_temp = settings["temp_range"]
            
            # Ensure current temperature is within allowed range for the slider
            current_temp = float(st.session_state.zones_data["temperature"][zone_index])
            if current_temp < min_temp:
                current_temp = min_temp
            elif current_temp > max_temp:
                current_temp = max_temp
                
            # Use a key specific to this zone and mode to avoid conflicts
            temp = st.slider(
                "Temperature", 
                min_value=min_temp, 
                max_value=max_temp, 
                value=current_temp, 
                step=0.1,
                key=f"temp_slider_{zone_index}_{mode}"
            )
            
            # Update session state with the new temperature
            st.session_state.zones_data["temperature"][zone_index] = float(temp)
            
            # Update energy usage based on temperature, mode, and weather
            energy_factor = settings["energy_factor"]
            weather_factor = 1 + (abs(weather_data['temperature'] - temp) * 0.02)
            humidity_factor = 1 + (weather_data['humidity'] - 50) * 0.005
            st.session_state.zones_data["energy_usage"][zone_index] = temp * 10 * energy_factor * weather_factor * humidity_factor
            
            st.session_state.zones_data["airflow"][zone_index] = st.radio(
                "Airflow", 
                ["Auto", "High", "Low"], 
                key=f"airflow_{zone_index}"
            )
        else:
            # Handle AUTO mode
            optimal_temp = predictor.predict_optimal_temperature(
                weather_data, 
                st.session_state.zones_data["occupancy"][zone_index],
                datetime.datetime.now().hour,
                datetime.datetime.now().weekday()
            )
            st.session_state.zones_data["temperature"][zone_index] = optimal_temp
            
            # Calculate energy usage based on optimal temperature and weather data
            weather_factor = 1 + (abs(weather_data['temperature'] - optimal_temp) * 0.02)
            humidity_factor = 1 + (weather_data['humidity'] - 50) * 0.005
            st.session_state.zones_data["energy_usage"][zone_index] = optimal_temp * 10 * weather_factor * humidity_factor
            
            st.session_state.zones_data["airflow"][zone_index] = "Auto"
            
            st.info(f"AI has set the optimal temperature to {optimal_temp:.1f}°C based on current conditions.")

def display_eco_points():
    points = st.session_state.eco_points
    badge_level = 0
    for level in sorted(BADGES.keys()):
        if points >= level:
            badge_level = level
    
    st.sidebar.markdown(f"""<div class='weather-card'>
        <h4>Your Eco Status</h4>
        <p>Points: {points}</p>
        <p>Badge: {BADGES[badge_level]}</p>
        <p>Next badge at {next((level for level in sorted(BADGES.keys()) if level > points), points+50)} points</p>
    </div>""", unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Smart HVAC Control System", layout="wide")
    st.markdown(custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Create services that should be available throughout the app
    weather_service = WeatherService(WEATHER_API_KEY)
    hvac_predictor = HVACPredictor()
    alert = Alert()

    # Get weather data once so it's available in all sections
    city = st.sidebar.selectbox("Select City", INDIAN_CITIES, index=0)
    weather_data = weather_service.get_weather(city)
    
    # Check if city has changed, if so update all zones
    if st.session_state.current_city != city:
        st.session_state.current_city = city
        for zone_index in range(len(ZONES)):
            update_zone_based_on_weather(zone_index, weather_data, hvac_predictor)
    
    # Display weather info in sidebar regardless of section
    st.sidebar.markdown(f"""<div class='weather-card'>
        <h4>Weather in {city}</h4>
        <p>Temperature: {weather_data['temperature']}°C</p>
        <p>Humidity: {weather_data['humidity']}%</p>
        <p>Wind Speed: {weather_data['wind_speed']} m/s</p>
        <p>Condition: {weather_data['condition']}</p>
    </div>""", unsafe_allow_html=True)
    
    # Display eco points
    display_eco_points()
    
    sidebar_section = st.sidebar.radio("Select a Section", ["Location Settings", "Energy Usage", "Zone Control"])
    
    if sidebar_section == "Location Settings":
        st.markdown("<h1 style='color: #00ff00;'>🌬️ AI-Powered Comfort Meets Sustainability</h1>", unsafe_allow_html=True)
        st.markdown("""
            <div style='background-color: #2C2C44; padding: 20px; border-radius: 15px; margin-bottom: 30px;'>
                <p style='font-size: 18px; color: white;'>
                    Welcome to the <strong>Smart HVAC Control System</strong> – your intelligent companion for maintaining optimal indoor climates.
                    Using real-time weather data and AI-driven predictions, this tool adjusts each room's settings to ensure both comfort and energy efficiency.
                    Make eco-friendly choices and earn points & badges as you reduce your energy footprint 🌱.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.title("🏠 Smart HVAC Zone Control")
        
        # Display a summary of all zones
        st.markdown("<h2 class='section-header'>Zone Summary</h2>", unsafe_allow_html=True)
        cols = st.columns(3)
        for i, zone in enumerate(ZONES):
            with cols[i]:
                st.markdown(f"""<div class='control-card'>
                    <h3>{zone}</h3>
                    <p>Temperature: {st.session_state.zones_data['temperature'][i]:.1f}°C</p>
                    <p>Mode: {st.session_state.zones_data['mode'][i]}</p>
                    <p>Energy: {st.session_state.zones_data['energy_usage'][i]:.1f} kWh</p>
                    <p>Humidity: {st.session_state.zones_data['humidity'][i]:.1f}%</p>
                </div>""", unsafe_allow_html=True)

    elif sidebar_section == "Energy Usage":
        st.subheader("Energy Usage Monitoring")
        alert.check_alerts(st.session_state.zones_data)
        alert.display_alerts()

        for zone_index in range(3):
            zone = ZONES[zone_index]
            st.markdown(f"### {zone} Energy Usage")
            
            # Generate unique data for each zone using city-specific weather data
            zone_temp = weather_data['temperature'] + (zone_index - 1) * 2
            zone_humidity = weather_data['humidity'] + (zone_index - 1) * 5
            
            # Apply zone-specific energy factor based on current mode
            mode = st.session_state.zones_data["mode"][zone_index]
            energy_factor = ZONE_MODES[mode]["energy_factor"]
            
            # Use a new seed that combines zone_index and city to ensure different data for each city
            seed_value = hash(f"{zone_index}_{city}") % 10000
            zone_data = generate_energy_data_with_weather(zone_temp, zone_humidity, seed=seed_value)
            zone_data['Usage'] = zone_data['Usage'] * energy_factor
            
            fig = plot_energy_usage(zone_data, zone_name=zone)
            st.pyplot(fig)
            
            # Display weather impact statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weather Impact", f"{abs(weather_data['temperature'] - 22) * 2:.1f}% change", 
                          delta=f"{'+' if weather_data['temperature'] > 22 else '-'}{abs(weather_data['temperature'] - 22):.1f}°C from baseline")
            with col2:
                st.metric("Humidity Impact", f"{abs(weather_data['humidity'] - 50) * 0.5:.1f}% change",
                          delta=f"{'+' if weather_data['humidity'] > 50 else '-'}{abs(weather_data['humidity'] - 50):.1f}% from baseline")

    elif sidebar_section == "Zone Control":
        st.subheader("Control Your Zones")
        for zone_index in range(3):
            display_zone_controls(zone_index, weather_data, hvac_predictor)

# Add this line at the end of your script
if __name__ == "__main__":
    main()