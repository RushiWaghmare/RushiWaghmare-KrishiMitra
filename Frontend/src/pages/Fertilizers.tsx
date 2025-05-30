
import { useState, useEffect } from 'react';
import { useLanguage } from '../contexts/LanguageContext';
import { cropsData } from '../data/cropsData';
import { CropData } from '../types/crop';
import SearchForm from '../components/fertilizers/SearchForm';
import WelcomeScreen from '../components/fertilizers/WelcomeScreen';
import CropDetail from '../components/fertilizers/CropDetail';

const Fertilizers = () => {
  const { t } = useLanguage();
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredCrop, setFilteredCrop] = useState<CropData | null>(null);

  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredCrop(null);
      return;
    }

    const foundCrop = cropsData.find(crop => 
      crop.name.en.toLowerCase().includes(searchTerm.toLowerCase()) || 
      crop.name.mr.toLowerCase().includes(searchTerm.toLowerCase())
    );

    setFilteredCrop(foundCrop || null);
  }, [searchTerm]);

  return (
    <div className="container mx-10 py-8 ">
      <h1 className="text-3xl font-bold mb-4 text-farm-green-dark">
        {t("fertilizers.title")}
      </h1>
      <p className="text-gray-700 mb-8 max-w-3xl">
        {t("fertilizers.description")}
      </p>

      <div className="mb-10">
        <SearchForm searchTerm={searchTerm} setSearchTerm={setSearchTerm} />
      </div>

      {!filteredCrop && searchTerm.trim() !== '' && (
        <div className="text-center py-8">
          <p className="text-gray-600">{t("fertilizers.noResults")}</p>
        </div>
      )}

      {filteredCrop && (
        <div className="space-y-10">
          <CropDetail crop={filteredCrop} />
        </div>
      )}

      {/* {!filteredCrop && searchTerm.trim() === '' && (
        <WelcomeScreen searchTerm={searchTerm} setSearchTerm={setSearchTerm} />
      )} */}
    </div>
  );
};

export default Fertilizers;
