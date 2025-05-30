
import React from 'react';
import MainLayout from '@/layouts/MainLayout';
import Hero from '@/components/Hero';
import FeaturedProducts from '@/components/FeaturedProducts';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Index = () => {
  return (
    <MainLayout>
      <Hero />
      <FeaturedProducts />
      
      <section className="container py-16 text-center">
        <h2 className="text-3xl font-bold mb-4">Ready to explore fresh produce directly from farmers?</h2>
        <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
          Join thousands of farmers and customers across India who are building a more sustainable food system.
        </p>
        <div className="flex flex-col sm:flex-row justify-center gap-4">
          <Button asChild size="lg">
            <Link to="/products">Explore Products</Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link to="/list-product">List Your Product</Link>
          </Button>
        </div>
      </section>
    </MainLayout>
  );
};

export default Index;
