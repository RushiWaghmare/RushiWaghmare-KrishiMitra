
import React from 'react';
import { Search } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useLanguage } from '../../contexts/LanguageContext';

interface SearchFormProps {
  searchTerm: string;
  setSearchTerm: (term: string) => void;
}

const SearchForm = ({ searchTerm, setSearchTerm }: SearchFormProps) => {
  const { t } = useLanguage();

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
  };

  return (
    <form onSubmit={handleSearch} className="flex gap-2 max-w-md">
      <div className="relative flex-grow">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={18} />
        <Input
          type="text"
          placeholder={t("fertilizers.searchPlaceholder")}
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-farm-green w-full"
        />
      </div>
      <Button type="submit" className="bg-farm-green hover:bg-farm-green-dark text-white">
        {t("fertilizers.search")}
      </Button>
    </form>
  );
};

export default SearchForm;
