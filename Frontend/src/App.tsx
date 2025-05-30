
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Crops from "./pages/Crops";
import Schemes from "./pages/Schemes";
import Fertilizers from "./pages/Fertilizers";
import NotFound from "./pages/NotFound";
import Navbar from "./components/Navbar";
import DiseaseDetection from "./pages/Disease";

import CropPricePrediction from "./pages/CropPricePrediction";
import { LanguageProvider } from "./contexts/LanguageContext";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <LanguageProvider>
        <div className="min-h-screen bg-gradient-to-br from-farm-gray/30 to-farm-beige/30">
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <Navbar />
            <div className="container mx-auto px-4 py-8">
              <Routes>
                <Route path="/" element={<Index />} />
                <Route path="/crops" element={<Crops />} />
                <Route path="/schemes" element={<Schemes />} />
                <Route path="/fertilizers" element={<Fertilizers />} />
                <Route path="/disease" element={<DiseaseDetection />} />
                
                <Route path="/croppriceprediction" element={<CropPricePrediction />} />
                {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
          </BrowserRouter>
        </div>
      </LanguageProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
