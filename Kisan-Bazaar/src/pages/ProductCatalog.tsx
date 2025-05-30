
import React, { useState, useEffect } from 'react';
import MainLayout from '@/layouts/MainLayout';
import ProductCard, { Product } from '@/components/ProductCard';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Link } from 'react-router-dom';
import { Search, Filter, Loader2 } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Sample products data (In a real app, this would come from an API)
const sampleProducts: Product[] = [
  {
    id: 1,
    name: "Organic Rice",
    price: 85,
    unit: "kg",
    quantity: 500,
    category: "Grains",
    description: "Premium quality organic rice grown without pesticides. Harvested last month and ready for delivery anywhere in Punjab and nearby regions.",
    region: "Nagpur",  // changed from Punjab
    contactName: "Gurpreet Singh",
    contactPhone: "9876543210",
    contactEmail: "gurpreet@example.com",
    contactWhatsapp: "9876543210",
    dateAdded: "2023-04-01T10:30:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  },
  {
    id: 2,
    name: "Fresh Tomatoes",
    price: 40,
    unit: "kg",
    quantity: 100,
    category: "Vegetables",
    description: "Fresh, ripe tomatoes harvested this week. Perfect for making sauce, curry or salad.",
    region: "Nashik",  // changed from Karnataka
    contactName: "Meena Kumari",
    contactPhone: "8765432109",
    contactWhatsapp: "8765432109",
    dateAdded: "2023-04-02T14:45:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  },
  {
    id: 3,
    name: "Alphonso Mangoes",
    price: 450,
    unit: "dozen",
    quantity: 50,
    category: "Fruits",
    description: "Premium Alphonso mangoes from Ratnagiri. Sweet, aromatic and perfect ripeness.",
    region: "Ratnagiri",  // already a city in Maharashtra
    contactName: "Rajesh Patil",
    contactPhone: "7654321098",
    contactEmail: "rajesh@example.com",
    dateAdded: "2023-04-03T09:15:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  },
  {
    id: 4,
    name: "Fresh Turmeric",
    price: 75,
    unit: "kg",
    quantity: 200,
    category: "Spices",
    description: "Organic turmeric roots, freshly harvested. High curcumin content and excellent for health benefits.",
    region: "Kolhapur",  // changed from Kerala
    contactName: "Priya Nair",
    contactPhone: "6543210987",
    contactEmail: "priya@example.com",
    contactWhatsapp: "6543210987",
    dateAdded: "2023-04-04T16:20:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  },
  {
    id: 5,
    name: "Organic Wheat",
    price: 45,
    unit: "kg",
    quantity: 1000,
    category: "Grains",
    description: "Premium quality organic wheat grown with traditional farming methods. No chemicals used.",
    region: "Aurangabad",  // changed from Madhya Pradesh
    contactName: "Ramesh Patel",
    contactPhone: "9876543210",
    contactWhatsapp: "9876543210",
    dateAdded: "2023-04-05T11:30:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  },
  {
    id: 6,
    name: "Farm Fresh Milk",
    price: 60,
    unit: "liter",
    quantity: 50,
    category: "Dairy",
    description: "Fresh cow milk from our farm, delivered daily. Pasteurized and ready to consume.",
    region: "Pune",  // changed from Gujarat
    contactName: "Anand Patel",
    contactPhone: "8765432109",
    contactEmail: "anand@example.com",
    dateAdded: "2023-04-06T07:45:00",
    image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
  }
];


const ProductCatalog = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("all");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [isLoading, setIsLoading] = useState(false);
  const [allProducts, setAllProducts] = useState<Product[]>([]);
  
  // Load products on component mount
  useEffect(() => {
    // Get user-added products from local storage
    const listedProducts = JSON.parse(localStorage.getItem('listedProducts') || '[]');
    
    // Combine sample products with user-added products
    setAllProducts([...sampleProducts, ...listedProducts]);
  }, []);
  
  // Extract unique regions and categories from all products
  const regions = [...new Set(allProducts.map(p => p.region))];
  const categories = [...new Set(allProducts.map(p => p.category))];

  // Filter products based on search term, region, and category
  const filteredProducts = allProducts.filter(product => {
    const matchesSearch = product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                        (product.description && product.description.toLowerCase().includes(searchTerm.toLowerCase())) ||
                        product.contactName.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRegion = selectedRegion === "all" || product.region === selectedRegion;
    const matchesCategory = selectedCategory === "all" || product.category === selectedCategory;
    
    return matchesSearch && matchesRegion && matchesCategory;
  });

  // Simulate search loading
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate API call delay
    setTimeout(() => {
      setIsLoading(false);
    }, 500);
  };

  return (
    <MainLayout>
      <div className="bg-muted py-8">
        <div className="container">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h1 className="text-3xl font-bold mb-2">Product Catalog</h1>
              <p className="text-muted-foreground">
                Browse fresh produce directly from farmers across India
              </p>
            </div>
            <Button asChild size="lg" className="w-full md:w-auto">
              <Link to="/list-product">
                List Your Product
              </Link>
            </Button>
          </div>
        </div>
      </div>
      
      <div className="container py-8">
        <div className="bg-white p-4 rounded-lg border mb-8">
          <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-grow">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Search products or farmers..."
                className="pl-8 w-full"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <Select value={selectedRegion} onValueChange={setSelectedRegion}>
              <SelectTrigger className="w-full md:w-[180px]">
                <SelectValue placeholder="All Regions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Regions</SelectItem>
                {regions.map(region => (
                  <SelectItem key={region} value={region}>{region}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Select value={selectedCategory} onValueChange={setSelectedCategory}>
              <SelectTrigger className="w-full md:w-[180px]">
                <SelectValue placeholder="All Categories" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Categories</SelectItem>
                {categories.map(category => (
                  <SelectItem key={category} value={category}>{category}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            
            <Button type="submit" className="md:w-auto" disabled={isLoading}>
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Filter className="mr-2 h-4 w-4" />
                  Filter
                </>
              )}
            </Button>
          </form>
        </div>
        
        <div className="flex items-center justify-between mb-6">
          <p className="text-muted-foreground">
            Showing {filteredProducts.length} products
          </p>
          
          <Button 
            variant="outline" 
            onClick={() => {
              setSearchTerm("");
              setSelectedRegion("all");
              setSelectedCategory("all");
            }}
          >
            Reset Filters
          </Button>
        </div>
        
        {filteredProducts.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProducts.map(product => (
              <ProductCard key={product.id} product={product} />
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <p className="text-lg font-medium">No products found matching your criteria</p>
            <p className="text-muted-foreground mt-2">Try adjusting your filters or search term</p>
          </div>
        )}
      </div>
    </MainLayout>
  );
};

export default ProductCatalog;
