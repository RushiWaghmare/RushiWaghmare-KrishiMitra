
import { useState, useEffect } from "react";
import { toast } from "@/hooks/use-toast";

export interface WeatherData {
  location: string;
  temperature: number;
  description: string;
  humidity: number;
  rainfall: string;
  forecast: {
    day: string;
    temp: number;
    description: string;
  }[];
  isLoading: boolean;
  error: string | null;
}

// OpenWeatherMap API key
const WEATHER_API_KEY = "8d2de98e089f1c28e1a22fc19a24ef04";

export const useWeather = () => {
  const [weatherData, setWeatherData] = useState<WeatherData>({
    location: "Loading...",
    temperature: 0,
    description: "Loading...",
    humidity: 0,
    rainfall: "0 mm",
    forecast: [],
    isLoading: true,
    error: null
  });

  useEffect(() => {
    const fetchWeatherData = async (lat: number, lon: number) => {
      try {
        // Get location name using reverse geocoding API from OpenWeatherMap
        const geocodingResponse = await fetch(
          `https://api.openweathermap.org/geo/1.0/reverse?lat=${lat}&lon=${lon}&limit=1&appid=${WEATHER_API_KEY}`
        );
        
        if (!geocodingResponse.ok) {
          throw new Error("Failed to fetch location data");
        }
        
        const locationData = await geocodingResponse.json();
        let locationName = "Unknown Location";
        
        if (locationData && locationData.length > 0) {
          const place = locationData[0];
          locationName = place.name + (place.state ? `, ${place.state}` : "") + (place.country ? `, ${place.country}` : "");
        }
        
        // Current weather
        const currentResponse = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${WEATHER_API_KEY}`
        );
        
        if (!currentResponse.ok) {
          throw new Error("Failed to fetch current weather data");
        }
        
        const currentData = await currentResponse.json();
        
        // 5-day forecast
        const forecastResponse = await fetch(
          `https://api.openweathermap.org/data/2.5/forecast?lat=${lat}&lon=${lon}&units=metric&appid=${WEATHER_API_KEY}`
        );
        
        if (!forecastResponse.ok) {
          throw new Error("Failed to fetch forecast data");
        }
        
        const forecastData = await forecastResponse.json();
        
        // Process forecast data (get one entry per day)
        const dailyForecasts = forecastData.list
          .filter((_: any, index: number) => index % 8 === 0) // Get one reading per day (every 24h)
          .slice(0, 5) // Get 5 days
          .map((item: any) => {
            const date = new Date(item.dt * 1000);
            const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
            return {
              day: dayNames[date.getDay()],
              temp: Math.round(item.main.temp),
              description: item.weather[0].main
            };
          });

        // Update state with all data
        setWeatherData({
          location: locationName, // Use more precise location from reverse geocoding
          temperature: Math.round(currentData.main.temp),
          description: currentData.weather[0].main,
          humidity: currentData.main.humidity,
          rainfall: currentData.rain ? `${currentData.rain["1h"] || 0} mm` : "0 mm",
          forecast: dailyForecasts,
          isLoading: false,
          error: null
        });
      } catch (error) {
        console.error("Error fetching weather:", error);
        setWeatherData(prev => ({ 
          ...prev, 
          isLoading: false, 
          error: "Failed to fetch weather data" 
        }));
        toast({
          title: "Weather Error",
          description: "Could not retrieve weather information. Please try again later.",
          variant: "destructive"
        });
      }
    };

    const handleGeoSuccess = (position: GeolocationPosition) => {
      const { latitude, longitude } = position.coords;
      fetchWeatherData(latitude, longitude);
    };

    const handleGeoError = (error: GeolocationPositionError) => {
      console.error("Geolocation error:", error);
      setWeatherData(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: "Unable to access your location" 
      }));
      toast({
        title: "Location Error",
        description: "Please allow location access to get accurate weather data.",
        variant: "destructive"
      });
    };

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(handleGeoSuccess, handleGeoError, {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0
      });
    } else {
      setWeatherData(prev => ({ 
        ...prev, 
        isLoading: false, 
        error: "Geolocation is not supported by this browser" 
      }));
      toast({
        title: "Browser Error",
        description: "Your browser doesn't support geolocation.",
        variant: "destructive"
      });
    }
  }, []);

  return weatherData;
};
