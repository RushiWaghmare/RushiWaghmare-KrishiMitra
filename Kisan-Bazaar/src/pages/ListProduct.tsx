
import React from 'react';
import MainLayout from '@/layouts/MainLayout';
import ListProductForm from '@/components/ListProductForm';

const ListProduct = () => {
  return (
    <MainLayout>
      <div className="bg-muted py-8">
        <div className="container">
          <h1 className="text-3xl font-bold mb-2">List Your Product</h1>
          <p className="text-muted-foreground">
            Connect directly with customers by listing your farm produce
          </p>
        </div>
      </div>
      
      <div className="container py-8">
        <ListProductForm />
      </div>
    </MainLayout>
  );
};

export default ListProduct;
