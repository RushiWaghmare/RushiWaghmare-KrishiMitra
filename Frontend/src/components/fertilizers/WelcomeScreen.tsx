
import React from 'react';
import { Beaker, Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { useLanguage } from '../../contexts/LanguageContext';

interface WelcomeScreenProps {
  searchTerm: string;
  setSearchTerm: (term: string) => void;
}

const WelcomeScreen = ({ searchTerm, setSearchTerm }: WelcomeScreenProps) => {
  const { t, language } = useLanguage();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-3xl mx-auto">
      <div className="text-center flex flex-col items-center">
        <Beaker size={60} className="text-farm-green-dark mb-4" />
        <h2 className="text-xl font-semibold mb-2">{t("fertilizers.search")}</h2>
        <p className="text-gray-600 mb-4">{t("fertilizers.description")}</p>
        <div className="w-full max-w-md">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
            <Input
              type="text"
              placeholder={t("fertilizers.searchPlaceholder")}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-farm-green w-full"
            />
          </div>
        </div>
        <div className="mt-6 text-sm text-gray-500">
          <p>Try searching for: {language === 'en' ? 'Rice, Wheat, Cotton' : 'तांदूळ, गहू, कापूस'}</p>
        </div>
      </div>
    </div>
  );
};

export default WelcomeScreen;
