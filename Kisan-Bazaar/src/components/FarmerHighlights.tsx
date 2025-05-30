
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Link } from "react-router-dom";
import { MapPin, Phone, Mail } from "lucide-react";

const farmers = [
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
  }
];

const FarmerHighlights = () => {
  return (
    <div className="bg-muted py-12">
      <div className="container">
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8">
          <div>
            <h2 className="text-3xl font-bold">Meet Our Farmers</h2>
            <p className="text-muted-foreground mt-2">The passionate people behind your food</p>
          </div>
          <Button asChild variant="outline" className="mt-4 md:mt-0">
            <Link to="/farmers">View All Farmers</Link>
          </Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {farmers.map(farmer => (
            <Card key={farmer.id} className="overflow-hidden">
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
      </div>
    </div>
  );
};

export default FarmerHighlights;
