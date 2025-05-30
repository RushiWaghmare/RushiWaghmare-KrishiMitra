// Simulated ARIMA model predictions
export const generatePredictions = (
    historicalData: number[],
    numPredictions: number = 90 // Changed to 90 days (3 months)
  ): number[] => {
    const predictions: number[] = [];
    const lastValue = historicalData[historicalData.length - 1];
    
    // Enhanced ARIMA simulation with seasonal patterns
    for (let i = 0; i < numPredictions; i++) {
      // Random variation (±5%)
      const randomFactor = Math.random() * 0.1 - 0.05;
      
      // Seasonal component (3-month cycle)
      const seasonalFactor = Math.sin((i / 90) * 2 * Math.PI) * 0.08;
      
      // Trend component (slight upward trend)
      const trend = 0.001 * i;
      
      // Market volatility (increases with time)
      const volatility = (1 + (i / numPredictions) * 0.05) * randomFactor;
      
      const newPrediction = lastValue * (1 + volatility + seasonalFactor + trend);
      predictions.push(Math.round(newPrediction));
    }
    
    return predictions;
  };
  
  // Generate dummy historical data
  export const generateHistoricalData = (
    startDate: Date,
    numPoints: number = 60, // Increased to 60 days of historical data
    basePrice: number,
    region: string
  ): { dates: string[]; prices: number[] } => {
    const dates: string[] = [];
    const prices: number[] = [];
    
    for (let i = 0; i < numPoints; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(currentDate.getDate() + i);
      dates.push(currentDate.toISOString().split('T')[0]);
      
      // Enhanced historical data generation
      const randomFactor = Math.random() * 0.2 - 0.1; // Random variation ±10%
      const seasonalFactor = Math.sin((i / 90) * 2 * Math.PI) * 0.08; // 3-month seasonal cycle
      const trend = 0.0005 * i; // Slight historical trend
      const price = basePrice * (1 + randomFactor + seasonalFactor + trend);
      prices.push(Math.round(price));
    }
    
    return { dates, prices };
  };