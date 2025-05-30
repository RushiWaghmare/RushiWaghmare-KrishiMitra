
import { Truck, Users, ShoppingBag, Leaf } from 'lucide-react';

const steps = [
  {
    icon: <Users className="h-10 w-10 text-kisan-600" />,
    title: "Connect Directly",
    description: "Farmers and customers connect without intermediaries, ensuring fair prices."
  },
  {
    icon: <Leaf className="h-10 w-10 text-kisan-600" />,
    title: "Fresh Produce",
    description: "Get farm-fresh products directly from the source, harvested at peak freshness."
  },
  {
    icon: <ShoppingBag className="h-10 w-10 text-kisan-600" />,
    title: "Shop Regionally",
    description: "Support local farmers in your region and reduce carbon footprint."
  },
  {
    icon: <Truck className="h-10 w-10 text-kisan-600" />,
    title: "Convenient Delivery",
    description: "Arrange pickup or delivery directly with farmers in your area."
  }
];

const HowItWorks = () => {
  return (
    <div className="container py-16">
      <h2 className="text-3xl font-bold text-center mb-12">How Kisan Bazaar Works</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        {steps.map((step, index) => (
          <div 
            key={index} 
            className="flex flex-col items-center text-center p-6 rounded-lg bg-white border border-border hover:shadow-md transition-all"
          >
            <div className="mb-4 p-3 rounded-full bg-accent">
              {step.icon}
            </div>
            <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
            <p className="text-muted-foreground">{step.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default HowItWorks;
