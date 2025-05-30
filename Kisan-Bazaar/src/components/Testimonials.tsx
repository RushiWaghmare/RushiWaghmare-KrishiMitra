
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const testimonials = [
  {
    content: "Kisan Bazaar has transformed how I sell my produce. I can now reach customers directly, offer fair prices, and get paid immediately.",
    author: {
      name: "Ramesh Yadav",
      role: "Wheat Farmer, Uttar Pradesh",
      avatar: "RY"
    }
  },
  {
    content: "I love knowing exactly where my food comes from. The quality of produce I get from Kisan Bazaar is incredible, and I feel good supporting local farmers.",
    author: {
      name: "Anita Sharma",
      role: "Customer, Delhi",
      avatar: "AS"
    }
  },
  {
    content: "My family has been farming for generations, but finding reliable buyers was always a challenge. This platform has connected me with customers who appreciate quality produce.",
    author: {
      name: "Govind Patel",
      role: "Vegetable Farmer, Gujarat",
      avatar: "GP"
    }
  }
];

const Testimonials = () => {
  return (
    <div className="bg-kisan-50 py-16">
      <div className="container">
        <h2 className="text-3xl font-bold text-center mb-12">What People Say</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {testimonials.map((testimonial, index) => (
            <Card key={index} className="border-none shadow-sm bg-white">
              <CardHeader className="pb-2">
                <div className="flex items-center space-x-1">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <svg
                      key={star}
                      className="h-5 w-5 fill-kisan-500"
                      viewBox="0 0 20 20"
                      fill="currentColor"
                    >
                      <path
                        fillRule="evenodd"
                        d="M10.868 2.884c-.321-.772-1.415-.772-1.736 0l-1.83 4.401-4.753.381c-.833.067-1.171 1.107-.536 1.651l3.62 3.102-1.106 4.637c-.194.813.691 1.456 1.405 1.02L10 15.591l4.069 2.485c.713.436 1.598-.207 1.404-1.02l-1.106-4.637 3.62-3.102c.635-.544.297-1.584-.536-1.65l-4.752-.382-1.831-4.401z"
                        clipRule="evenodd"
                      />
                    </svg>
                  ))}
                </div>
              </CardHeader>
              <CardContent className="pb-4">
                <p className="text-muted-foreground">"{testimonial.content}"</p>
              </CardContent>
              <CardFooter>
                <div className="flex items-center space-x-4">
                  <Avatar>
                    <AvatarFallback className="bg-kisan-100 text-kisan-800">
                      {testimonial.author.avatar}
                    </AvatarFallback>
                  </Avatar>
                  <div>
                    <div className="text-sm font-medium">{testimonial.author.name}</div>
                    <div className="text-xs text-muted-foreground">{testimonial.author.role}</div>
                  </div>
                </div>
              </CardFooter>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Testimonials;
