
import { Button } from "@/components/ui/button";
import { useLocation } from "react-router-dom";
import { useEffect } from "react";
import { Link } from "react-router-dom";
import MainLayout from "@/layouts/MainLayout";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <MainLayout>
      <div className="container flex flex-col items-center justify-center py-24 text-center">
        <h1 className="text-9xl font-bold text-kisan-500">404</h1>
        <p className="text-2xl font-semibold mt-6 mb-2">Page Not Found</p>
        <p className="text-muted-foreground mb-8 max-w-md">
          We couldn't find the page you were looking for. It might have been moved or deleted.
        </p>
        <Button asChild size="lg">
          <Link to="/">Return to Home</Link>
        </Button>
      </div>
    </MainLayout>
  );
};

export default NotFound;
