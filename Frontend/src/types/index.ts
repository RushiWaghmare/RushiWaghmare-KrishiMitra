export interface CropData {
    date: string;
    price: number;
    crop: string;
  }
  
  export interface Prediction {
    date: string;
    predictedPrice: number;
    confidenceInterval: {
      lower: number;
      upper: number;
    };
  }