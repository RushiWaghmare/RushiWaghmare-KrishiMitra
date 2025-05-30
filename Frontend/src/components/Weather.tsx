
import { Sun, CloudRain, MapPin, Cloud, CloudSnow, CloudLightning, CloudFog, Loader2 } from "lucide-react";
import { useWeather } from "../hooks/useWeather";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useLanguage } from "../contexts/LanguageContext";

const Weather = () => {
  const weather = useWeather();
  const { t } = useLanguage();

  const getWeatherIcon = (description: string) => {
    const desc = description.toLowerCase();
    if (desc.includes("rain") || desc.includes("drizzle")) {
      return <CloudRain className="h-12 w-12 text-blue-500 mr-2" />;
    } else if (desc.includes("cloud")) {
      return <Cloud className="h-12 w-12 text-gray-500 mr-2" />;
    } else if (desc.includes("snow")) {
      return <CloudSnow className="h-12 w-12 text-blue-200 mr-2" />;
    } else if (desc.includes("thunder") || desc.includes("lightning")) {
      return <CloudLightning className="h-12 w-12 text-yellow-500 mr-2" />;
    } else if (desc.includes("fog") || desc.includes("mist")) {
      return <CloudFog className="h-12 w-12 text-gray-400 mr-2" />;
    } else {
      return <Sun className="h-12 w-12 text-yellow-500 mr-2" />;
    }
  };

  if (weather.isLoading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-8 flex items-center justify-center">
        <Loader2 className="h-8 w-8 text-farm-green animate-spin" />
        <span className="ml-2 text-gray-600">{t("weather.loading")}</span>
      </div>
    );
  }

  if (weather.error) {
    return (
      <Alert variant="destructive" className="bg-white rounded-lg shadow-md overflow-hidden">
        <AlertDescription>
          {weather.error}. {t("weather.fallback")}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      <div className="p-4 bg-farm-green text-white">
        <h3 className="font-bold text-xl flex items-center">
          <MapPin className="mr-2 h-5 w-5" />
          {t("weather.in")} {weather.location}
        </h3>
      </div>
      <div className="p-4">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            {getWeatherIcon(weather.description)}
            <div>
              <p className="text-3xl font-bold">{weather.temperature}°C</p>
              <p className="text-gray-600">{weather.description}</p>
            </div>
          </div>
          <div>
            <div className="text-right">
              <p className="text-sm text-gray-600">{t("weather.humidity")} {weather.humidity}%</p>
              <p className="text-sm text-gray-600">{t("weather.rainfall")} {weather.rainfall}</p>
            </div>
          </div>
        </div>
        
        <div className="border-t pt-4">
          <h4 className="font-semibold text-gray-700 mb-2">{t("weather.forecast")}</h4>
          {weather.forecast.length > 0 ? (
            <div className="grid grid-cols-5 gap-2">
              {weather.forecast.map((day) => (
                <div key={day.day} className="text-center">
                  <p className="font-medium">{day.day}</p>
                  <p className="text-xl font-bold">{day.temp}°</p>
                  <p className="text-xs text-gray-600">{day.description}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-500">{t("weather.unavailable")}</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default Weather;
