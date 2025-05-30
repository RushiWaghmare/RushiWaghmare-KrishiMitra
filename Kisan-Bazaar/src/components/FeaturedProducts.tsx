import React, { useState, useEffect } from 'react';
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ShoppingCart, Clock, ArrowRight } from "lucide-react";
import { Link } from 'react-router-dom';

const sampleProducts = [
  {
    id: 1,
    name: "Red Apples",
    price: 120,
    unit: "kg",
    farmer: "Raj Singh",
    region: "Nashik",
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
    image: "https://images.unsplash.com/photo-1601493700631-2b16ec4b4716?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    available: false
  }
];


const FeaturedProducts = () => {
  const [allProducts, setAllProducts] = useState([...sampleProducts]);
  
  useEffect(() => {
    const listedProducts = JSON.parse(localStorage.getItem('listedProducts') || '[]');
    const userProducts = listedProducts.slice(0, 2);
    setAllProducts([...sampleProducts, ...userProducts]);
  }, []);

  return (
    <div className="container py-12">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
        <div>
          <h2 className="text-3xl font-bold">Featured Products</h2>
          <p className="text-muted-foreground mt-2">Fresh picks from farms across India</p>
        </div>
        <Button asChild variant="outline" className="mt-4 md:mt-0">
          <Link to="/products">
            View All Products
            <ArrowRight className="ml-2 h-4 w-4" />
          </Link>
        </Button>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {allProducts.map(product => (
          <Card key={product.id} className="overflow-hidden flex flex-col h-full transition-all hover:shadow-md">
            <div className="relative aspect-square w-full overflow-hidden">
              <img 
                src={product.image} 
                alt={product.name} 
                className="w-full h-full object-cover transition-transform hover:scale-105 duration-500"
              />
              {!product.available && product.available !== undefined && (
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
              <CardTitle className="flex justify-between items-start">
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
                asChild
                className="w-full"
              >
                <Link to="/products">
                  <ShoppingCart className="mr-2 h-4 w-4" />
                  View Product
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default FeaturedProducts;
