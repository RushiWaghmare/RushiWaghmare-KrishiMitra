
import { Button } from "@/components/ui/button";
import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";

const Hero = () => {
  return (
    <div className="relative overflow-hidden bg-gradient-to-br from-kisan-900 to-kisan-700 text-white">
      <div className="pattern-bg absolute inset-0 opacity-10"></div>
      <div className="container relative py-16 md:py-24">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div className="flex flex-col space-y-6">
            <div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight mb-4 text-shadow">
                Farm Fresh, <br />
                <span className="text-kisan-300">Direct to You</span>
              </h1>
              <p className="text-lg md:text-xl max-w-md text-white/90">
                Connect directly with farmers across India and get fresh produce delivered to your doorstep.
              </p>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild size="lg" className="bg-kisan-500 hover:bg-kisan-600 text-white">
                <Link to="/products">
                  Explore Products
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="border-white/40 text-white hover:bg-white/10">
                <Link to="/list-product">List Your Product</Link>
              </Button>
            </div>
            
            <div className="flex items-center gap-x-6 text-sm">
              <div className="flex -space-x-2">
                {[1, 2, 3, 4].map((i) => (
                  <div 
                    key={i} 
                    className="h-8 w-8 rounded-full border-2 border-kisan-700 bg-kisan-100 flex items-center justify-center text-kisan-700 font-bold text-xs"
                  >
                    KB
                  </div>
                ))}
              </div>
              <div className="text-white/80">
                <span className="font-semibold text-white">2,000+</span> farmers across India
              </div>
            </div>
          </div>
          
          <div className="relative hidden md:block">
            <div className="absolute -top-8 -right-8 w-20 h-20 bg-kisan-300 rounded-full opacity-30"></div>
            <div className="absolute -bottom-10 -left-10 w-32 h-32 bg-kisan-400 rounded-full opacity-30"></div>
            
            <div className="relative z-10 grid grid-cols-2 gap-4">
              <div className="space-y-4">
                <div className="rounded-lg overflow-hidden h-48">
                  <img 
                    src="https://images.unsplash.com/photo-1469041797191-50ace28483c3?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Indian Farmer" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="rounded-lg overflow-hidden h-32">
                  <img 
                    src="https://images.unsplash.com/photo-1452378174528-3090a4bba7b2?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Rural Farm" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="space-y-4 pt-6">
                <div className="rounded-lg overflow-hidden h-32">
                  <img 
                    src="https://images.unsplash.com/photo-1485833077593-4278bba3f11f?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Fresh Produce" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="rounded-lg overflow-hidden h-48">
                  <img 
                    src="https://images.unsplash.com/photo-1465379944081-7f47de8d74ac?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Farm Animals" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="bg-gradient-to-r from-soil-500 to-soil-600 py-4">
        <div className="container">
          <div className="flex flex-wrap justify-center md:justify-between gap-4 items-center text-white">
            <p className="text-center md:text-left font-medium">
              🌾 Support Indian farmers and get fresh, seasonal produce directly from the source
            </p>
            <Button asChild variant="secondary" className="bg-white hover:bg-white/90 text-soil-800">
              <Link to="/products">Explore Products</Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
