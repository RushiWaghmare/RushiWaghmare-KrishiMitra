
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const regions = [
  {
    id: "pune",
    name: "Pune",
    description: "A major educational and IT hub known for its pleasant climate and cultural heritage.",
    image: "https://images.unsplash.com/photo-1580906855287-cc11cfc43c17?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80", 
    products: ["Sugarcane", "Pomegranate", "Grapes", "Turmeric"]
  },
  {
    id: "mumbai",
    name: "Mumbai",
    description: "The capital city of Maharashtra and the financial hub of India, known for its bustling ports and urban agriculture.",
    image: "https://images.unsplash.com/photo-1585241936936-753c60f7d2f6?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    products: ["Vegetables", "Flowers", "Coconut", "Rice"]
  },
  {
    id: "nagpur",
    name: "Nagpur",
    description: "Known as the Orange City, located in central India with rich agricultural output.",
    image: "https://images.unsplash.com/photo-1562686831-2ef40ca2069f?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    products: ["Oranges", "Soybeans", "Wheat", "Cotton"]
  },
  {
    id: "nashik",
    name: "Nashik",
    description: "Famous for vineyards and temples, Nashik is a key agricultural region in Maharashtra.",
    image: "https://images.unsplash.com/photo-1610440043049-58c1951c9c48?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    products: ["Grapes", "Onions", "Tomatoes", "Pulses"]
  },
  {
    id: "aurangabad",
    name: "Aurangabad",
    description: "A historic city known for its silk industry and proximity to Ajanta and Ellora caves.",
    image: "https://images.unsplash.com/photo-1606118311367-49dc2beae4eb?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80",
    products: ["Cotton", "Bajra", "Jowar", "Tur"]
  }
];


const RegionSelector = () => {
  return (
    <div className="container py-12">
      <h2 className="text-3xl font-bold text-center mb-8">Shop by Region</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {regions.map((region) => (
          <Card key={region.id} className="overflow-hidden transition-all hover:shadow-md">
            <div className="aspect-video w-full overflow-hidden">
              <img 
                src={region.image} 
                alt={region.name} 
                className="w-full h-full object-cover transition-transform hover:scale-105 duration-500"
              />
            </div>
            <CardHeader>
              <CardTitle>{region.name}</CardTitle>
              <CardDescription>{region.description}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex flex-wrap gap-2">
                {region.products.map(product => (
                  <span 
                    key={product} 
                    className="px-2 py-1 bg-accent text-accent-foreground text-xs rounded-full"
                  >
                    {product}
                  </span>
                ))}
              </div>
            </CardContent>
            <CardFooter>
              <Button asChild className="w-full">
                <Link to={`/region/${region.id}`}>
                  Explore Region
                </Link>
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default RegionSelector;
