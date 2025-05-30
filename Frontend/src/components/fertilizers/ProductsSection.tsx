
import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Product } from '../../types/crop';
import { useLanguage } from '../../contexts/LanguageContext';
import ProductCard from './ProductCard';

interface ProductsSectionProps {
  title: string;
  badgeClass: string;
  products: Product[];
}

const ProductsSection = ({ title, badgeClass, products }: ProductsSectionProps) => {
  const [showAll, setShowAll] = useState(false);
  const { t } = useLanguage();
  
  const displayedProducts = showAll ? products : products.slice(0, 2);

  return (
    <div className="mb-8">
      <h4 className="text-lg font-medium mb-4 flex items-center">
        <span className={`${badgeClass} text-xs font-medium me-2 px-2.5 py-0.5 rounded`}>
          {title}
        </span>
      </h4>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {displayedProducts.map((product) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
      
      {products.length > 2 && (
        <div className="mt-4 text-center">
          <Button 
            variant="outline" 
            onClick={() => setShowAll(!showAll)}
            className="border-farm-green text-farm-green hover:bg-farm-green/10"
          >
            {showAll ? t("crop.showLess") : t("fertilizers.loadMore")}
          </Button>
        </div>
      )}
    </div>
  );
};

export default ProductsSection;
