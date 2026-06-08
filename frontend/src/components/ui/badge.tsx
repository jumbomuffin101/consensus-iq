import * as React from "react";

import { cn } from "@/lib/utils";

type BadgeProps = React.HTMLAttributes<HTMLDivElement> & {
  tone?: "default" | "success" | "warning" | "danger" | "muted";
};

const tones = {
  default: "border-transparent bg-primary text-primary-foreground",
  success: "border-emerald-500/30 bg-emerald-500/15 text-emerald-200",
  warning: "border-amber-500/30 bg-amber-500/15 text-amber-200",
  danger: "border-red-500/30 bg-red-500/15 text-red-200",
  muted: "border-border bg-muted text-muted-foreground",
};

export function Badge({ className, tone = "default", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold",
        tones[tone],
        className,
      )}
      {...props}
    />
  );
}
