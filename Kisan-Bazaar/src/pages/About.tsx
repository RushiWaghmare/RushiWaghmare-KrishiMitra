
import React from 'react';
import MainLayout from '@/layouts/MainLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Link } from 'react-router-dom';
import { BadgeIndianRupee, Users, ShoppingBag, Truck, Phone, Mail, MapPin } from 'lucide-react';

const About = () => {
  return (
    <MainLayout>
      <div className="bg-muted py-8">
        <div className="container">
          <h1 className="text-3xl font-bold mb-2">About Kisan Bazaar</h1>
          <p className="text-muted-foreground">Our mission and vision for Indian agriculture</p>
        </div>
      </div>
      
      <div className="container py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div>
            <h2 className="text-2xl font-bold mb-4">Our Story</h2>
            <p className="text-muted-foreground mb-4">
              Kisan Bazaar was founded in 2023 with a simple but powerful mission: to create a direct connection between farmers and consumers across India. We believe that farmers deserve fair prices for their produce, and consumers deserve access to fresh, high-quality food.
            </p>
            <p className="text-muted-foreground mb-4">
              Our team comes from diverse backgrounds in agriculture, technology, and retail, but we share a common passion for supporting Indian farmers and promoting sustainable food systems.
            </p>
            <p className="text-muted-foreground mb-6">
              Today, Kisan Bazaar connects thousands of farmers with customers in every region of India, enabling transparent, direct transactions that benefit everyone involved.
            </p>
            <Button asChild>
              <Link to="/register">Join Our Community</Link>
            </Button>
          </div>
          
          <div className="relative">
            <div className="absolute -z-10 top-0 -left-4 w-72 h-72 bg-green-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
            <div className="absolute -z-10 top-0 -right-4 w-72 h-72 bg-yellow-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
            <div className="absolute -z-10 -bottom-8 left-20 w-72 h-72 bg-rose-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
            
            <div className="grid grid-cols-2 gap-4 relative">
              <div className="space-y-4">
                <div className="rounded-lg overflow-hidden h-48 shadow-md">
                  <img 
                    src="https://images.unsplash.com/photo-1472396961693-142e6e269027?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Indian Farmer" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="rounded-lg overflow-hidden h-32 shadow-md">
                  <img 
                    src="https://images.unsplash.com/photo-1493962853295-0fd70327578a?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Rural Farm" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
              <div className="space-y-4 pt-6">
                <div className="rounded-lg overflow-hidden h-32 shadow-md">
                  <img 
                    src="https://images.unsplash.com/photo-1465379944081-7f47de8d74ac?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Fresh Produce" 
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="rounded-lg overflow-hidden h-48 shadow-md">
                  <img 
                    src="https://images.unsplash.com/photo-1466721591366-2d5fba72006d?ixlib=rb-1.2.1&auto=format&fit=crop&w=600&q=80" 
                    alt="Farm Animals" 
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="my-16">
          <h2 className="text-2xl font-bold text-center mb-12">Our Values</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="bg-accent border-none">
              <CardContent className="pt-6">
                <div className="rounded-full w-12 h-12 bg-kisan-100 flex items-center justify-center mb-4">
                  <BadgeIndianRupee className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Fair Prices</h3>
                <p className="text-sm text-muted-foreground">
                  We ensure farmers receive fair compensation for their hard work and consumers pay reasonable prices.
                </p>
              </CardContent>
            </Card>
            
            <Card className="bg-accent border-none">
              <CardContent className="pt-6">
                <div className="rounded-full w-12 h-12 bg-kisan-100 flex items-center justify-center mb-4">
                  <Users className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Community</h3>
                <p className="text-sm text-muted-foreground">
                  We're building a community that values transparency, trust, and mutual respect.
                </p>
              </CardContent>
            </Card>
            
            <Card className="bg-accent border-none">
              <CardContent className="pt-6">
                <div className="rounded-full w-12 h-12 bg-kisan-100 flex items-center justify-center mb-4">
                  <ShoppingBag className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Quality</h3>
                <p className="text-sm text-muted-foreground">
                  We promote high-quality produce that meets rigorous standards for freshness and safety.
                </p>
              </CardContent>
            </Card>
            
            <Card className="bg-accent border-none">
              <CardContent className="pt-6">
                <div className="rounded-full w-12 h-12 bg-kisan-100 flex items-center justify-center mb-4">
                  <Truck className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Sustainability</h3>
                <p className="text-sm text-muted-foreground">
                  We support sustainable farming practices and efficient local distribution networks.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
        
        <div className="mt-16">
          <h2 className="text-2xl font-bold text-center mb-4">Get in Touch</h2>
          <p className="text-center text-muted-foreground max-w-2xl mx-auto mb-8">
            Have questions about Kisan Bazaar? Want to learn more about our platform or how to get involved? Our team is here to help!
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <Card>
              <CardContent className="flex flex-col items-center text-center pt-6">
                <div className="rounded-full w-12 h-12 bg-accent flex items-center justify-center mb-4">
                  <Phone className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-medium mb-2">Phone</h3>
                <p className="text-muted-foreground">+91 98765 43210</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="flex flex-col items-center text-center pt-6">
                <div className="rounded-full w-12 h-12 bg-accent flex items-center justify-center mb-4">
                  <Mail className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-medium mb-2">Email</h3>
                <p className="text-muted-foreground">support@kisanbazaar.in</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="flex flex-col items-center text-center pt-6">
                <div className="rounded-full w-12 h-12 bg-accent flex items-center justify-center mb-4">
                  <MapPin className="text-kisan-600 h-6 w-6" />
                </div>
                <h3 className="font-medium mb-2">Address</h3>
                <p className="text-muted-foreground">123 Agritech Park, Bangalore 560001</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default About;
