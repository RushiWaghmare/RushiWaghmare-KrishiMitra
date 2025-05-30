
import { Link } from "react-router-dom";
import { useLanguage } from "../contexts/LanguageContext";

const Footer = () => {
  const { t } = useLanguage();
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="bg-farm-green text-white">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-bold mb-4">KrishiMitra</h3>
            <p className="text-farm-green-light text-sm">
              {t("footer.about")}
            </p>
          </div>
          <div>
            <h3 className="text-xl font-bold mb-4">{t("footer.quickLinks")}</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-farm-green-light hover:text-white transition">
                  {t("footer.home")}
                </Link>
              </li>
              <li>
                <Link to="/crops" className="text-farm-green-light hover:text-white transition">
                  {t("footer.cropDatabase")}
                </Link>
              </li>
              <li>
                <Link to="/schemes" className="text-farm-green-light hover:text-white transition">
                  {t("footer.govtSchemes")}
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-bold mb-4">{t("footer.contact")}</h3>
            <p className="text-farm-green-light text-sm">
              {t("footer.contactText")}
              <br />
              <a href="mailto:contact@farminfo.com" className="hover:text-white transition">contact@KrishiMitraInfo.com</a>
            </p>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-farm-green-light text-center">
          <p className="text-sm text-farm-green-light">
            © {currentYear} KrishiMitra {t("footer.rights")}
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
