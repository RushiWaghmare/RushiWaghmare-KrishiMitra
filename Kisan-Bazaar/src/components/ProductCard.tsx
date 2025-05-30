
import React from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Phone, Mail, MessageSquare } from 'lucide-react';

export type Product = {
  id: number;
  name: string;
  price: number;
  unit: string;
  quantity: number;
  category: string;
  description: string;
  region: string;
  contactName: string;
  contactPhone: string;
  contactEmail?: string;
  contactWhatsapp?: string;
  dateAdded: string;
  image?: string;
};

interface ProductCardProps {
  product: Product;
}

const ProductCard = ({ product }: ProductCardProps) => {
  const handlePhoneClick = () => {
    window.location.href = `tel:${product.contactPhone}`;
  };

  const handleEmailClick = () => {
    if (product.contactEmail) {
      window.location.href = `mailto:${product.contactEmail}?subject=Inquiry about ${product.name}`;
    }
  };

  const handleWhatsAppClick = () => {
    if (product.contactWhatsapp) {
      window.open(`https://wa.me/${product.contactWhatsapp.replace(/\D/g, '')}?text=I'm interested in your ${product.name} listing on Kisan Bazaar`, '_blank');
    }
  };

  return (
    <Card className="overflow-hidden flex flex-col h-full transition-all hover:shadow-md">
      {product.image ? (
        <div className="relative aspect-square w-full overflow-hidden">
          <img 
            src={product.image} 
            alt={product.name} 
            className="w-full h-full object-cover transition-transform hover:scale-105 duration-500"
          />
          <Badge className="absolute top-2 right-2 bg-white text-foreground">
            {product.region}
          </Badge>
        </div>
      ) : (
        <div className="relative aspect-square w-full overflow-hidden bg-muted flex items-center justify-center">
          <span className="text-2xl text-muted-foreground">{product.category}</span>
          <Badge className="absolute top-2 right-2 bg-white text-foreground">
            {product.region}
          </Badge>
        </div>
      )}
      
      <CardHeader>
        <div className="flex items-center justify-between">
          <Badge variant="outline">{product.category}</Badge>
          <span className="text-sm text-muted-foreground">
            {new Date(product.dateAdded).toLocaleDateString()}
          </span>
        </div>
        <CardTitle className="flex justify-between items-start mt-2">
          <span>{product.name}</span>
          <span className="text-kisan-600">
            ₹{product.price}/{product.unit}
          </span>
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          by {product.contactName}
        </p>
      </CardHeader>
      
      <CardContent>
        <p className="text-sm mb-2">
          <span className="font-medium">Available:</span> {product.quantity} {product.unit}
        </p>
        <p className="text-sm line-clamp-3">
          {product.description}
        </p>
      </CardContent>
      
      <CardFooter className="mt-auto flex flex-col gap-2">
        <div className="grid grid-cols-3 gap-2 w-full">
          <Button 
            onClick={handlePhoneClick}
            variant="outline" 
            className="flex items-center justify-center gap-1"
            title="Call"
          >
            <Phone className="h-4 w-4" />
          </Button>
          
          <Button 
            onClick={handleEmailClick}
            variant="outline" 
            className="flex items-center justify-center gap-1"
            disabled={!product.contactEmail}
            title={product.contactEmail ? "Email" : "No email provided"}
          >
            <Mail className="h-4 w-4" />
          </Button>
          
          <Button 
            onClick={handleWhatsAppClick}
            variant="outline" 
            className="flex items-center justify-center gap-1"
            disabled={!product.contactWhatsapp}
            title={product.contactWhatsapp ? "WhatsApp" : "No WhatsApp provided"}
          >
            <MessageSquare className="h-4 w-4" />
          </Button>
        </div>
      </CardFooter>
    </Card>
  );
};

export default ProductCard;
