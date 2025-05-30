
import { useLanguage } from "../contexts/LanguageContext";

export type SchemeType = {
  id: number;
  title: string;
  ministry: string;
  eligibility: string;
  benefits: string;
  applicationProcess: string;
  lastDate?: string;
  website: string;
};

const SchemeCard = ({ scheme }: { scheme: SchemeType }) => {
  const { t } = useLanguage();
  
  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden border-l-4 border-farm-green">
      <div className="p-5">
        <div className="flex justify-between items-start">
          <h3 className="text-xl font-bold text-farm-green mb-2">{scheme.title}</h3>
          {scheme.lastDate && (
            <span className="bg-red-100 text-red-800 text-xs font-medium px-2.5 py-0.5 rounded">
              {t("scheme.lastDate")} {scheme.lastDate}
            </span>
          )}
        </div>
        
        <p className="text-sm text-gray-500 mb-4">{t("scheme.ministry")} {scheme.ministry}</p>
        
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-1">{t("scheme.eligibility")}</h4>
          <p className="text-sm text-gray-600">{scheme.eligibility}</p>
        </div>
        
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-1">{t("scheme.benefits")}</h4>
          <p className="text-sm text-gray-600">{scheme.benefits}</p>
        </div>
        
        <div className="mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-1">{t("scheme.applicationProcess")}</h4>
          <p className="text-sm text-gray-600">{scheme.applicationProcess}</p>
        </div>
        
        <a
          href={scheme.website}
          target="_blank"
          rel="noopener noreferrer"
          className="bg-farm-green text-white text-sm font-medium py-2 px-4 rounded inline-block hover:bg-farm-green-dark transition"
        >
          {t("scheme.website")}
        </a>
      </div>
    </div>
  );
};

export default SchemeCard;
