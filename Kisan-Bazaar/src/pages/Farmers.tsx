
import React, { useState } from 'react';
import MainLayout from '@/layouts/MainLayout';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Link } from 'react-router-dom';
import { MapPin, Phone, Mail, Search } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// Sample farmers data
const allFarmers = [
  {
    id: 1,
    name: "Raj Singh",
    location: "Himachal Pradesh",
    image: "https://images.unsplash.com/photo-1520052203542-d3095f1b6cf0?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Apples", "Pears", "Walnuts"],
    description: "Third-generation apple farmer with organic certification.",
    phone: "+91 98765 12345",
    email: "raj@kisanbazaar.in"
  },
  {
    id: 2,
    name: "Meera Patel",
    location: "Punjab",
    image: "https://images.unsplash.com/photo-1590523741831-ab7e8b8f9c7f?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Wheat", "Rice", "Mustard"],
    description: "Agricultural cooperative leader specializing in sustainable rice farming.",
    phone: "+91 97654 23456",
    email: "meera@kisanbazaar.in"
  },
  {
    id: 3,
    name: "Priya Desai",
    location: "Maharashtra",
    image: "https://images.unsplash.com/photo-1509506489701-dfe23b067808?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Mangoes", "Grapes", "Pomegranate"],
    description: "Award-winning mango producer with exports to multiple countries.",
    phone: "+91 96543 34567",
    email: "priya@kisanbazaar.in"
  },
  {
    id: 4,
    name: "Arun Kumar",
    location: "Karnataka",
    image: "https://images.unsplash.com/photo-1597424216809-3ba9864aae1d?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Coffee", "Spices", "Vegetables"],
    description: "Coffee plantation owner practicing sustainable farming for over 15 years.",
    phone: "+91 95432 45678",
    email: "arun@kisanbazaar.in"
  },
  {
    id: 5,
    name: "Lakshmi Nair",
    location: "Kerala",
    image: "https://images.unsplash.com/photo-1523824921871-d6f1a15151f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Coconut", "Spices", "Banana"],
    description: "Spice farmer focusing on traditional cultivation methods and fair trade practices.",
    phone: "+91 94321 56789",
    email: "lakshmi@kisanbazaar.in"
  },
  {
    id: 6,
    name: "Gurpreet Singh",
    location: "Uttar Pradesh",
    image: "https://images.unsplash.com/photo-1592151450377-1159cc0f349e?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    specialties: ["Potatoes", "Onions", "Garlic"],
    description: "Organic vegetable grower supplying to premium markets across North India.",
    phone: "+91 93210 67890",
    email: "gurpreet@kisanbazaar.in"
  }
];

// Extract unique regions and specialties
const regions = [...new Set(allFarmers.map(f => f.location))];
const allSpecialties = allFarmers.flatMap(f => f.specialties);
const specialties = [...new Set(allSpecialties)];

const Farmers = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("");
  const [selectedSpecialty, setSelectedSpecialty] = useState("");

  // Filter farmers based on search term, region, and specialty
  const filteredFarmers = allFarmers.filter(farmer => {
    const matchesSearch = farmer.name.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRegion = selectedRegion === "" || farmer.location === selectedRegion;
    const matchesSpecialty = selectedSpecialty === "" || farmer.specialties.includes(selectedSpecialty);
    
    return matchesSearch && matchesRegion && matchesSpecialty;
  });

  return (
    <MainLayout>
      <div className="bg-muted py-8">
        <div className="container">
          <h1 className="text-3xl font-bold mb-2">Our Farmers</h1>
          <p className="text-muted-foreground">Meet the passionate people behind your food</p>
        </div>
      </div>
      
      <div className="container py-8">
        <div className="flex flex-col md:flex-row gap-6">
          {/* Filters sidebar */}
          <div className="w-full md:w-1/4">
            <div className="bg-white p-4 rounded-lg border mb-4">
              <h3 className="font-semibold mb-4">Search</h3>
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search farmers..."
                  className="pl-8"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
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
              <h3 className="font-semibold mb-4">Filter by Specialty</h3>
              <Select value={selectedSpecialty} onValueChange={setSelectedSpecialty}>
                <SelectTrigger>
                  <SelectValue placeholder="All Specialties" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">All Specialties</SelectItem>
                  {specialties.map(specialty => (
                    <SelectItem key={specialty} value={specialty}>{specialty}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <Button variant="outline" 
              className="w-full"
              onClick={() => {
                setSearchTerm("");
                setSelectedRegion("");
                setSelectedSpecialty("");
              }}
            >
              Reset Filters
            </Button>
          </div>
          
          {/* Farmers grid */}
          <div className="w-full md:w-3/4">
            <div className="flex justify-between items-center mb-6">
              <p className="text-muted-foreground">
                Showing {filteredFarmers.length} farmers
              </p>
            </div>
            
            {filteredFarmers.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredFarmers.map(farmer => (
                  <Card key={farmer.id} className="overflow-hidden h-full">
                    <div className="aspect-[4/3] w-full overflow-hidden">
                      <img 
                        src={farmer.image} 
                        alt={farmer.name} 
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <CardHeader>
                      <div className="flex justify-between items-start">
                        <CardTitle>{farmer.name}</CardTitle>
                        <Badge variant="outline" className="flex items-center">
                          <MapPin className="h-3 w-3 mr-1" />
                          {farmer.location}
                        </Badge>
                      </div>
                      <CardDescription>{farmer.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold mb-2">Specialties:</h4>
                        <div className="flex flex-wrap gap-2">
                          {farmer.specialties.map(specialty => (
                            <Badge key={specialty}>{specialty}</Badge>
                          ))}
                        </div>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div className="flex items-center">
                          <Phone className="h-4 w-4 mr-2 text-muted-foreground" />
                          <span>{farmer.phone}</span>
                        </div>
                        <div className="flex items-center">
                          <Mail className="h-4 w-4 mr-2 text-muted-foreground" />
                          <span>{farmer.email}</span>
                        </div>
                      </div>
                    </CardContent>
                    <CardFooter>
                      <Button asChild className="w-full">
                        <Link to={`/farmers/${farmer.id}`}>View Profile</Link>
                      </Button>
                    </CardFooter>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <p className="text-lg font-medium">No farmers found matching your criteria</p>
                <p className="text-muted-foreground mt-2">Try adjusting your filters or search term</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default Farmers;
