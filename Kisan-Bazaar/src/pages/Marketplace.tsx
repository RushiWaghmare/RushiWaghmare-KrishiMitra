
import React, { useState } from 'react';
import MainLayout from '@/layouts/MainLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ShoppingCart, Clock, Filter, ChevronDown } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

// Sample products data
const allProducts = [
  {
    id: 1,
    name: "Red Apples",
    price: 120,
    unit: "kg",
    farmer: "Raj Singh",
    region: "Nashik",
    category: "Fruits",
    image: "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 2,
    name: "Organic Rice",
    price: 85,
    unit: "kg",
    farmer: "Meera Patel",
    region: "Kolhapur",
    category: "Grains",
    image: "https://images.unsplash.com/photo-1536304993881-ff6e9eefa2a6?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 3,
    name: "Fresh Tomatoes",
    price: 40,
    unit: "kg",
    farmer: "Arun Kumar",
    region: "Pune",
    category: "Vegetables",
    image: "https://images.unsplash.com/photo-1524593166156-312f362cada0?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 4,
    name: "Alphonso Mangoes",
    price: 450,
    unit: "dozen",
    farmer: "Priya Desai",
    region: "Ratnagiri",
    category: "Fruits",
    image: "https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: false
  },
  {
    id: 5,
    name: "Organic Potatoes",
    price: 30,
    unit: "kg",
    farmer: "Gurpreet Singh",
    region: "Solapur",
    category: "Vegetables",
    image: "https://images.unsplash.com/photo-1518977676601-b53f82aba655?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 6,
    name: "Fresh Turmeric",
    price: 75,
    unit: "kg",
    farmer: "Lakshmi Nair",
    region: "Nagpur",
    category: "Spices",
    image: "https://images.unsplash.com/photo-1615485500704-8e990f9925c9?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 7,
    name: "Organic Wheat",
    price: 45,
    unit: "kg",
    farmer: "Harpreet Singh",
    region: "Aurangabad",
    category: "Grains",
    image: "https://images.unsplash.com/photo-1565647952915-45709a2554f5?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 8,
    name: "Green Chillies",
    price: 60,
    unit: "kg",
    farmer: "Ravi Kumar",
    region: "Amravati",
    category: "Vegetables",
    image: "https://images.unsplash.com/photo-1588252303782-cb80119abd6d?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  },
  {
    id: 9,
    name: "Cardamom",
    price: 1200,
    unit: "kg",
    farmer: "Maya Thomas",
    region: "Satara",
    category: "Spices",
    image: "https://images.unsplash.com/photo-1638910160865-b547d36a0703?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: true
  }
];


// Extract unique regions and categories
const regions = [...new Set(allProducts.map(p => p.region))];
const categories = [...new Set(allProducts.map(p => p.category))];

const Marketplace = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [showAvailableOnly, setShowAvailableOnly] = useState(false);
  const [sortBy, setSortBy] = useState("name");

  // Filter products based on search term, region, and category
  const filteredProducts = allProducts.filter(product => {
    const matchesSearch = product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                        product.farmer.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRegion = selectedRegion === "" || product.region === selectedRegion;
    const matchesCategory = selectedCategory === "" || product.category === selectedCategory;
    const matchesAvailability = !showAvailableOnly || product.available;
    
    return matchesSearch && matchesRegion && matchesCategory && matchesAvailability;
  });

  // Sort products
  const sortedProducts = [...filteredProducts].sort((a, b) => {
    if (sortBy === "name") {
      return a.name.localeCompare(b.name);
    } else if (sortBy === "price-low") {
      return a.price - b.price;
    } else if (sortBy === "price-high") {
      return b.price - a.price;
    }
    return 0;
  });

  return (
    <MainLayout>
      <div className="bg-muted py-8">
        <div className="container">
          <h1 className="text-3xl font-bold mb-2">Marketplace</h1>
          <p className="text-muted-foreground">Browse fresh produce directly from farmers across India</p>
        </div>
      </div>
      
      <div className="container py-8">
        <div className="flex flex-col md:flex-row gap-6">
          {/* Filters sidebar */}
          <div className="w-full md:w-1/4">
            <div className="bg-white p-4 rounded-lg border mb-4">
              <h3 className="font-semibold mb-4">Search</h3>
              <Input
                placeholder="Search products or farmers..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="bg-white p-4 rounded-lg border mb-4">
              <h3 className="font-semibold mb-4">Filter by Region</h3>
              <Select value={selectedRegion} onValueChange={setSelectedRegion}>
                <SelectTrigger>
                  <SelectValue placeholder="All Regions" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Regions</SelectItem>
                  {regions.map(region => (
                    <SelectItem key={region} value={region}>{region}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="bg-white p-4 rounded-lg border mb-4">
              <h3 className="font-semibold mb-4">Filter by Category</h3>
              <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                <SelectTrigger>
                  <SelectValue placeholder="All Categories" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Categories</SelectItem>
                  {categories.map(category => (
                    <SelectItem key={category} value={category}>{category}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="bg-white p-4 rounded-lg border mb-4">
              <h3 className="font-semibold mb-4">Availability</h3>
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="available-only"
                  checked={showAvailableOnly}
                  onChange={() => setShowAvailableOnly(!showAvailableOnly)}
                  className="rounded text-kisan-600 focus:ring-kisan-500"
                />
                <label htmlFor="available-only">Show available products only</label>
              </div>
            </div>
            
            <Button variant="outline" 
              className="w-full"
              onClick={() => {
                setSearchTerm("");
                setSelectedRegion("");
                setSelectedCategory("");
                setShowAvailableOnly(false);
              }}
            >
              Reset Filters
            </Button>
          </div>
          
          {/* Products grid */}
          <div className="w-full md:w-3/4">
            <div className="flex justify-between items-center mb-6">
              <p className="text-muted-foreground">
                Showing {sortedProducts.length} products
              </p>
              
              <div className="flex items-center gap-2">
                <span className="text-sm">Sort by:</span>
                <Select value={sortBy} onValueChange={setSortBy}>
                  <SelectTrigger className="w-[160px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="name">Name (A-Z)</SelectItem>
                    <SelectItem value="price-low">Price (Low to High)</SelectItem>
                    <SelectItem value="price-high">Price (High to Low)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            
            {sortedProducts.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {sortedProducts.map(product => (
                  <Card key={product.id} className="overflow-hidden flex flex-col h-full transition-all hover:shadow-md">
                    <div className="relative aspect-square w-full overflow-hidden">
                      <img 
                        src={product.image} 
                        alt={product.name} 
                        className="w-full h-full object-cover transition-transform hover:scale-105 duration-500"
                      />
                      {!product.available && (
                        <div className="absolute inset-0 bg-background/80 flex items-center justify-center">
                          <Badge variant="secondary" className="text-lg py-2 px-4 bg-background">
                            <Clock className="w-4 h-4 mr-2" />
                            Coming Soon
                          </Badge>
                        </div>
                      )}
                      <Badge className="absolute top-2 right-2 bg-white text-foreground">
                        {product.region}
                      </Badge>
                    </div>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <Badge variant="outline">{product.category}</Badge>
                      </div>
                      <CardTitle className="flex justify-between items-start mt-2">
                        <span>{product.name}</span>
                        <span className="text-kisan-600">
                          ₹{product.price}/{product.unit}
                        </span>
                      </CardTitle>
                      <p className="text-sm text-muted-foreground">
                        by {product.farmer}
                      </p>
                    </CardHeader>
                    <CardFooter className="mt-auto">
                      <Button 
                        disabled={!product.available}
                        className="w-full"
                      >
                        <ShoppingCart className="mr-2 h-4 w-4" />
                        {product.available ? "Add to Cart" : "Not Available"}
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-lg font-medium">No products found matching your criteria</p>
                <p className="text-muted-foreground mt-2">Try adjusting your filters or search term</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default Marketplace;
