
import React, { useState } from 'react';
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { useNavigate } from 'react-router-dom';

// Form validation schema
const formSchema = z.object({
  productName: z.string().min(3, {
    message: "Product name must be at least 3 characters.",
  }),
  category: z.string({
    required_error: "Please select a product category.",
  }),
  price: z.string().refine((val) => !isNaN(Number(val)) && Number(val) > 0, {
    message: "Price must be a positive number.",
  }),
  quantity: z.string().refine((val) => !isNaN(Number(val)) && Number(val) >= 0, {
    message: "Quantity must be a non-negative number.",
  }),
  unit: z.string({
    required_error: "Please select a unit of measurement.",
  }),
  description: z.string().min(10, {
    message: "Description must be at least 10 characters.",
  }).max(500, {
    message: "Description must not exceed 500 characters."
  }),
  region: z.string({
    required_error: "Please select your region.",
  }),
  contactName: z.string().min(3, {
    message: "Contact name must be at least 3 characters.",
  }),
  contactPhone: z.string().min(10, {
    message: "Please enter a valid phone number.",
  }),
  contactEmail: z.string().email({
    message: "Please enter a valid email address.",
  }).optional(),
  contactWhatsapp: z.string().min(10, {
    message: "Please enter a valid WhatsApp number.",
  }).optional(),
});

const regions = [
  "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Solapur", "Amravati", 
  "Kolhapur", "Sangli", "Jalgaon", "Latur", "Satara", "Akola", 
  "Ahmednagar", "Chandrapur", "Dhule", "Beed", "Parbhani", "Nanded", "Ratnagiri", 
  "Wardha", "Yavatmal", "Bhiwandi", "Malegaon", "Panvel", "Thane", 
  "Miraj", "Osmanabad"
];


const categories = ["Fruits", "Vegetables", "Grains", "Spices", "Dairy", "Poultry", "Livestock", "Other"];
const units = ["kg", "gram", "dozen", "piece", "quintal", "ton", "liter"];

const ListProductForm = () => {
  const navigate = useNavigate();
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      productName: "",
      description: "",
      contactName: "",
      contactPhone: "",
    },
  });

  function onSubmit(values: z.infer<typeof formSchema>) {
    // Get existing products from local storage or initialize empty array
    const existingProducts = JSON.parse(localStorage.getItem('listedProducts') || '[]');
    
    // Create a new product object
    const newProduct = {
      id: Date.now(), // Use timestamp as a simple unique ID
      name: values.productName,
      price: Number(values.price),
      unit: values.unit,
      quantity: Number(values.quantity),
      category: values.category,
      description: values.description,
      region: values.region,
      contactName: values.contactName,
      contactPhone: values.contactPhone,
      contactEmail: values.contactEmail || undefined,
      contactWhatsapp: values.contactWhatsapp || undefined,
      dateAdded: new Date().toISOString(),
      image: "https://img.freepik.com/free-vector/shopping-supermarket-cart-with-grocery-pictogram_1284-11697.jpg"
    };
    
    // Add the new product to the array
    existingProducts.push(newProduct);
    
    // Save back to local storage
    localStorage.setItem('listedProducts', JSON.stringify(existingProducts));
    
    // Show success message
    toast.success("Product listed successfully!", {
      description: "Your product has been added to the marketplace.",
    });
    
    // Reset form
    form.reset();
    
    // Navigate to products page to see the newly added product
    setTimeout(() => {
      navigate('/products');
    }, 1500);
  }

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="bg-white p-6 rounded-lg border shadow-sm">
        <h2 className="text-2xl font-bold mb-6">List Your Product</h2>
        
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <FormField
                control={form.control}
                name="productName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Product Name*</FormLabel>
                    <FormControl>
                      <Input placeholder="e.g. Organic Tomatoes" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="category"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Category*</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a category" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {categories.map((category) => (
                          <SelectItem key={category} value={category}>
                            {category}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <FormField
                control={form.control}
                name="price"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Price (₹)*</FormLabel>
                    <FormControl>
                      <Input type="number" min="0" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="quantity"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Quantity*</FormLabel>
                    <FormControl>
                      <Input type="number" min="0" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              
              <FormField
                control={form.control}
                name="unit"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Unit*</FormLabel>
                    <Select onValueChange={field.onChange} defaultValue={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select unit" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {units.map((unit) => (
                          <SelectItem key={unit} value={unit}>
                            {unit}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
            
            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description*</FormLabel>
                  <FormControl>
                    <Textarea 
                      placeholder="Describe your product quality, freshness, harvesting date, etc."
                      className="min-h-[120px]" 
                      {...field} 
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <FormField
              control={form.control}
              name="region"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Region*</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select your region" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent className="max-h-[200px]">
                      {regions.map((region) => (
                        <SelectItem key={region} value={region}>
                          {region}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            
            <div className="border-t pt-6">
              <h3 className="text-lg font-semibold mb-4">Contact Information</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                  control={form.control}
                  name="contactName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Contact Name*</FormLabel>
                      <FormControl>
                        <Input placeholder="Your full name" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="contactPhone"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Phone Number*</FormLabel>
                      <FormControl>
                        <Input placeholder="Your phone number" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="contactEmail"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Email (Optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="Your email address" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                
                <FormField
                  control={form.control}
                  name="contactWhatsapp"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>WhatsApp (Optional)</FormLabel>
                      <FormControl>
                        <Input placeholder="Your WhatsApp number" {...field} />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>
            
            <div className="flex justify-end pt-4">
              <Button type="button" variant="outline" className="mr-4" onClick={() => form.reset()}>
                Reset
              </Button>
              <Button type="submit">Submit Listing</Button>
            </div>
          </form>
        </Form>
      </div>
    </div>
  );
};

export default ListProductForm;
