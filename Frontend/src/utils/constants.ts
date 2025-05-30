export const REGIONS = [
    'Andhra Pradesh',
    'Bihar',
    'Gujarat',
    'Haryana',
    'Karnataka',
    'Madhya Pradesh',
    'Maharashtra',
    'Punjab',
    'Rajasthan',
    'Tamil Nadu',
    'Uttar Pradesh',
    'West Bengal'
  ];
  
  // Base prices in rupees per quintal
  export const getBasePrice = (region: string) => {
    const regionalMultiplier: { [key: string]: number } = {
      'Punjab': 1.2,
      'Haryana': 1.15,
      'Uttar Pradesh': 1.1,
      'Maharashtra': 1.05,
      'Gujarat': 1.08,
      'Rajasthan': 0.95,
      'Madhya Pradesh': 0.98,
      'Bihar': 0.92,
      'West Bengal': 1.0,
      'Karnataka': 1.02,
      'Tamil Nadu': 1.05,
      'Andhra Pradesh': 1.03
    };
  
    return regionalMultiplier[region] || 1;
  };