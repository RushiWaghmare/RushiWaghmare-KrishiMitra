
export interface Product {
  id: string;
  name: {
    en: string;
    mr: string;
  };
  image: string;
  type: {
    en: string;
    mr: string;
  };
  usage: {
    en: string;
    mr: string;
  };
  benefits: {
    en: string[];
    mr: string[];
  };
  application: {
    en: string;
    mr: string;
  };
  price: string;
  link: string;
}

export interface CropData {
  id: string;
  name: {
    en: string;
    mr: string;
  };
  fertilizers: Product[];
  pesticides: Product[];
}
