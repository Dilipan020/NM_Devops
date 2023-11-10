from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
from datetime import datetime
import requests
import folium
from folium.plugins import MarkerCluster

app = Flask(__name__)

@app.route('/')
def index_view():
    return render_template('ind.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        filename = 'models/final_model.pkl'
        gbr = pickle.load(open(filename, 'rb'))
        date = request.form['date']
        time = request.form['time']
        holiday = request.form.get('holiday', '0')  # Get the value of 'holiday' key or default to '0'
        date_time_str = date + ' ' + time
        date_time = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M')
        day = int(date_time.day)
        day_of_week = int(date_time.weekday())
        hour = int(date_time.hour)
        minute = int(date_time.minute)
        hour_min = round(hour + (minute / 60), 1)

        # Create a DataFrame for user input
        user_input = pd.DataFrame({'day': [day], 'day_of_week': [day_of_week], 'hour_min': [hour_min], 'Holiday': [int(holiday)]})

        # Make prediction on user input using the trained model
        user_pred = gbr.predict(user_input)

        # Add your map logic here
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = """
        [out:json];
        area["ISO3166-1"="IN"];
        (node["amenity"="parking"](area);
         way["amenity"="parking"](area);
         rel["amenity"="parking"](area);
        );
        out center;
        """
        
        # Send a GET request to the Overpass API
        response = requests.get(overpass_url, params={'data': overpass_query})
        data = response.json()
        m = folium.Map(location=[12.9921, 80.2172], zoom_start=15, tiles='OpenStreetMap')
        location = [12.9921, 80.2172]
        # Create a MarkerCluster layer
        marker_cluster = MarkerCluster().add_to(m)
        icon_text = '<div style="font-size: 18px; color: Black; font-weight: bold; text-align: center;">BusyLot</div>'
        icon = folium.DivIcon(html=icon_text)
        icon2 = folium.Icon(icon="star", color="red")
        folium.Marker(location=location, icon=icon).add_to(m)
        folium.Marker(location=location, icon=icon2).add_to(m)
        # Iterate through the elements and add markers to the MarkerCluster layer
        for element in data['elements']:
            if element['type'] == 'node':
                lon = element['lon']
                lat = element['lat']
                coords = [lat, lon]
                folium.Marker(location=coords).add_to(marker_cluster)
            elif 'center' in element:
                lon = element['center']['lon']
                lat = element['center']['lat']
                coords = [lat, lon]
                folium.Marker(location=coords).add_to(marker_cluster)
        
        # Save the map as an image
        map_path = 'static/images/map.html'
        m.save(map_path)

        # Generate the response message based on the predicted output
        if user_pred[0] * 100 < 30:
            response = str(round(user_pred[0] * 100, 2)) + "% of spaces are occupied. The parking lot is free."
        elif user_pred[0] * 100 < 60:
            response = str(round(user_pred[0] * 100, 2)) + "% of spaces are occupied. The parking lot is a little busy."
        else:
            response = str(round(user_pred[0] * 100, 2)) + "% of spaces are occupied. The parking lot is busy, consider these alternatives."

        return redirect(url_for('prediction', prediction=response, current_date=date, current_time=time, url=map_path))

@app.route('/prediction')
def prediction():
    prediction = request.args.get('prediction')
    current_date = request.args.get('current_date')
    current_time = request.args.get('current_time')
    url = request.args.get('url')
    return render_template('prediction.html', prediction=prediction, current_date=current_date, current_time=current_time, url=url)

if __name__ == '__main__':
    app.run(debug=True)
